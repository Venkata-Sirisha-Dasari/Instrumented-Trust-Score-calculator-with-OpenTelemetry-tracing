"""Trust score evaluation engine for six AI governance dimensions.

This module contains the core scoring engine, validation, risk classification,
artifact generation, and SHA-256 sealing for audit evidence.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import jsonschema
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


# ---------------------------------------------------------------------------
# LOGGING  (structured, not just print statements — production habit)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("TrustScoreEngine")


# ---------------------------------------------------------------------------
# OPEN TELEMETRY TRACING
# ---------------------------------------------------------------------------

def setup_tracing() -> trace.Tracer:
    resource = Resource.create({"service.name": "trust-score-evaluator"})
    provider = TracerProvider(resource=resource)
    exporter = JaegerExporter(agent_host_name="localhost", agent_port=6831)
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


tracer = setup_tracing()


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
EVALUATOR_VERSION = "1.0.0"

VALID_DIMENSIONS = {
    "accuracy",
    "robustness",
    "fairness",
    "safety",
    "privacy",
    "transparency",
}

SCORE_MIN: float = 0.0
SCORE_MAX: float = 1.0
WEIGHT_MIN: float = 0.0

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "propertyNames": {"enum": sorted(VALID_DIMENSIONS)},
            "additionalProperties": {
                "type": "number",
                "minimum": SCORE_MIN,
                "maximum": SCORE_MAX,
            },
        },
        "weights": {
            "type": "object",
            "propertyNames": {"enum": sorted(VALID_DIMENSIONS)},
            "additionalProperties": {
                "type": "number",
                "minimum": WEIGHT_MIN,
            },
        },
    },
    "required": ["scores", "weights"],
    "additionalProperties": False,
}

FLOAT_TOLERANCE: float = 1e-9

RISK_THRESHOLDS = {
    "CRITICAL": 0.40,
    "HIGH":     0.60,
    "MEDIUM":   0.75,
    "LOW":      1.01,
}

DIMENSION_FLAG_THRESHOLDS = {
    "accuracy":     0.60,
    "robustness":   0.60,
    "fairness":     0.55,
    "safety":       0.50,
    "privacy":      0.55,
    "transparency": 0.55,
}

WEIGHT_LARGE_WARN: float = 10.0
SCORE_MAX_ALLOWED: float = 1e6


# ---------------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------------
class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class RiskFlag(str, Enum):
    ACCURACY_RISK     = "ACCURACY_RISK"
    ROBUSTNESS_RISK   = "ROBUSTNESS_RISK"
    FAIRNESS_RISK     = "FAIRNESS_RISK"
    SAFETY_RISK       = "SAFETY_RISK"
    PRIVACY_RISK      = "PRIVACY_RISK"
    TRANSPARENCY_RISK = "TRANSPARENCY_RISK"
    MISSING_DIMENSION = "MISSING_DIMENSION"
    WEIGHT_ANOMALY    = "WEIGHT_ANOMALY"


# ---------------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------------
class TrustScoreError(Exception):
    pass


class ValidationError(TrustScoreError):
    pass


class NormalizationError(TrustScoreError):
    pass


class EvidenceError(TrustScoreError):
    pass


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------
@dataclass
class TrustScoreResult:
    trust_score:         float
    risk_level:          RiskLevel
    risk_flags:          list[RiskFlag]
    scores:              dict[str, float]
    weights:             dict[str, float]
    normalized_weights:  dict[str, float]


@dataclass
class EvidenceArtifact:
    artifact_id:         str
    timestamp:           str
    evaluator_version:   str
    scores:              dict[str, float]
    weights:             dict[str, float]
    normalized_weights:  dict[str, float]
    trust_score:         float
    risk_level:          str
    risk_flags:          list[str]
    sha256_hash:         str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "artifact_id":        self.artifact_id,
            "timestamp":          self.timestamp,
            "evaluator_version":  self.evaluator_version,
            "scores":             self.scores,
            "weights":            self.weights,
            "normalized_weights": self.normalized_weights,
            "trust_score":        round(self.trust_score, 6),
            "risk_level":         self.risk_level,
            "risk_flags":         self.risk_flags,
        }

    def to_json(self, include_hash: bool = True) -> str:
        data = self.to_dict()
        if include_hash:
            data["sha256_hash"] = self.sha256_hash
        return json.dumps(data, sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# MAIN ENGINE
# ---------------------------------------------------------------------------
class TrustScoreCalculator:
    def __init__(self, scores: dict[str, float], weights: dict[str, float]) -> None:
        self._raw_scores = dict(scores)
        self._raw_weights = dict(weights)
        self.scores = dict(scores)
        self.weights = dict(weights)
        self.normalized_weights: dict[str, float] = {}
        self._warnings: list[str] = []
        self.last_evidence: EvidenceArtifact | None = None
        logger.info("TrustScoreCalculator initialised | dimensions=%s", list(scores.keys()))

    def validate_inputs(self) -> None:
        with tracer.start_as_current_span("input_validation") as span:
            span.set_attribute("input.score_count", len(self.scores))
            span.set_attribute("input.weight_count", len(self.weights))

            try:
                jsonschema.validate(
                    instance={"scores": self.scores, "weights": self.weights},
                    schema=INPUT_SCHEMA,
                )
            except jsonschema.ValidationError as exc:
                span.set_attribute("input.validation_error", True)
                raise ValidationError(f"Input JSON schema validation failed: {exc.message}") from exc

            if not self.scores:
                span.set_attribute("input.empty_scores", True)
                raise ValidationError("scores dict is empty.  At least one dimension is required.")
            if not self.weights:
                span.set_attribute("input.empty_weights", True)
                raise ValidationError("weights dict is empty.  At least one weight is required.")

            unknown = set(self.scores.keys()) - VALID_DIMENSIONS
            if unknown:
                span.set_attribute("input.unknown_dimensions", ",".join(sorted(unknown)))
                raise ValidationError(
                    f"Unknown trust dimension(s): {unknown}.  Valid dimensions: {VALID_DIMENSIONS}"
                )

            unknown_w = set(self.weights.keys()) - VALID_DIMENSIONS
            if unknown_w:
                span.set_attribute("input.unknown_weight_dimensions", ",".join(sorted(unknown_w)))
                raise ValidationError(
                    f"Unknown weight dimension(s): {unknown_w}.  Valid dimensions: {VALID_DIMENSIONS}"
                )

            for dim, score in self.scores.items():
                if not isinstance(score, (int, float)):
                    span.set_attribute("input.invalid_score_type", dim)
                    raise ValidationError(
                        f"Score for '{dim}' must be numeric, got {type(score).__name__}."
                    )
                if score > SCORE_MAX_ALLOWED:
                    span.set_attribute("input.score_too_large", float(score))
                    raise ValidationError(
                        f"Score for '{dim}' = {score} exceeds maximum allowed ({SCORE_MAX_ALLOWED})."
                    )
                if not (SCORE_MIN <= score <= SCORE_MAX):
                    span.set_attribute("input.score_out_of_range", float(score))
                    raise ValidationError(
                        f"Score for '{dim}' = {score} is outside valid range [{SCORE_MIN}, {SCORE_MAX}]."
                    )

            for dim, weight in self.weights.items():
                if not isinstance(weight, (int, float)):
                    span.set_attribute("input.invalid_weight_type", dim)
                    raise ValidationError(
                        f"Weight for '{dim}' must be numeric, got {type(weight).__name__}."
                    )
                if weight < WEIGHT_MIN:
                    span.set_attribute("input.negative_weight", float(weight))
                    raise ValidationError(
                        f"Weight for '{dim}' = {weight} is negative."
                    )
                if weight > WEIGHT_LARGE_WARN:
                    msg = (
                        f"Weight for '{dim}' = {weight} is unusually large."
                    )
                    logger.warning(msg)
                    self._warnings.append(msg)
                    span.set_attribute("input.large_weight_warning", dim)

            missing_dims = VALID_DIMENSIONS - set(self.scores.keys())
            if missing_dims:
                msg = (
                    f"Missing trust dimensions: {missing_dims}."
                )
                logger.warning(msg)
                self._warnings.append(msg)
                span.set_attribute("input.missing_dimensions", ",".join(sorted(missing_dims)))

            missing_weights = set(self.scores.keys()) - set(self.weights.keys())
            if missing_weights:
                msg = (
                    f"No weight provided for dimension(s): {missing_weights}."
                )
                logger.warning(msg)
                self._warnings.append(msg)
                span.set_attribute("input.missing_weights", ",".join(sorted(missing_weights)))
                for dim in missing_weights:
                    self.weights[dim] = 0.0

            span.set_attribute("input.valid", True)
            logger.info("Input validation passed.")

    def normalize_weights(self) -> dict[str, float]:
        with tracer.start_as_current_span("weight_normalization") as span:
            weight_sum = sum(self.weights.values())
            span.set_attribute("weight.sum_before", float(weight_sum))
            span.set_attribute("weight.count", len(self.weights))

            if abs(weight_sum) < FLOAT_TOLERANCE:
                span.set_attribute("weight.all_zero", True)
                raise NormalizationError(
                    "All weights are zero.  Cannot normalize."
                )

            self.normalized_weights = {
                dim: (w / weight_sum)
                for dim, w in self.weights.items()
            }

            check = sum(self.normalized_weights.values())
            span.set_attribute("weight.sum_after", float(check))
            span.set_attribute("weight.large_warning_count", len([w for w in self.weights.values() if w > WEIGHT_LARGE_WARN]))

            if abs(check - 1.0) > 1e-6:
                span.set_attribute("weight.normalization_error", float(check))
                raise NormalizationError(
                    f"Normalization produced weights summing to {check:.8f}."
                )

            logger.info("Weights normalised | sum_before=%.4f | sum_after=%.6f", weight_sum, check)
            return self.normalized_weights

    def calculate_score(self) -> TrustScoreResult:
        with tracer.start_as_current_span("trust_score_pipeline") as pipeline_span:
            pipeline_span.set_attribute("pipeline.dimensions", len(self.scores))
            self.validate_inputs()
            self.normalize_weights()

            with tracer.start_as_current_span("score_calculation") as span:
                span.set_attribute("score.dimension_count", len(self.scores))
                trust_score: float = 0.0
                for dim, score in self.scores.items():
                    w = self.normalized_weights.get(dim, 0.0)
                    trust_score += w * score
                    logger.debug(
                        "Dimension %-14s | score=%.4f | norm_weight=%.4f | contrib=%.4f",
                        dim, score, w, w * score,
                    )

                trust_score = max(SCORE_MIN, min(SCORE_MAX, trust_score))

                risk_level = self.classify_risk(trust_score)
                risk_flags = self.generate_risk_flags()

                span.set_attribute("score.trust_score", float(trust_score))
                span.set_attribute("score.risk_level", risk_level.value)
                span.set_attribute("score.risk_flags", ",".join([flag.value for flag in risk_flags]))

                logger.info(
                    "Trust score computed | score=%.4f | risk=%s | flags=%s",
                    trust_score, risk_level.value, [f.value for f in risk_flags],
                )

                result = TrustScoreResult(
                    trust_score=trust_score,
                    risk_level=risk_level,
                    risk_flags=risk_flags,
                    scores=dict(self.scores),
                    weights=dict(self._raw_weights),
                    normalized_weights=dict(self.normalized_weights),
                )

                self.last_evidence = self.generate_evidence(result)
                return result

    def classify_risk(self, trust_score: float) -> RiskLevel:
        if trust_score < RISK_THRESHOLDS["CRITICAL"]:
            return RiskLevel.CRITICAL
        elif trust_score <= RISK_THRESHOLDS["HIGH"]:
            return RiskLevel.HIGH
        elif trust_score < RISK_THRESHOLDS["MEDIUM"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def generate_risk_flags(self) -> list[RiskFlag]:
        flags: list[RiskFlag] = []
        dimension_flag_map: dict[str, RiskFlag] = {
            "accuracy":     RiskFlag.ACCURACY_RISK,
            "robustness":   RiskFlag.ROBUSTNESS_RISK,
            "fairness":     RiskFlag.FAIRNESS_RISK,
            "safety":       RiskFlag.SAFETY_RISK,
            "privacy":      RiskFlag.PRIVACY_RISK,
            "transparency": RiskFlag.TRANSPARENCY_RISK,
        }

        for dim, threshold in DIMENSION_FLAG_THRESHOLDS.items():
            if dim in self.scores and self.scores[dim] < threshold:
                flags.append(dimension_flag_map[dim])
                logger.warning(
                    "Risk flag raised | %s | score=%.4f | threshold=%.4f",
                    dimension_flag_map[dim].value,
                    self.scores[dim],
                    threshold,
                )

        missing = VALID_DIMENSIONS - set(self.scores.keys())
        if missing:
            flags.append(RiskFlag.MISSING_DIMENSION)

        return flags

    def generate_evidence(self, result: TrustScoreResult) -> EvidenceArtifact:
        with tracer.start_as_current_span("evidence_generation") as span:
            artifact = EvidenceArtifact(
                artifact_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                evaluator_version=EVALUATOR_VERSION,
                scores=result.scores,
                weights=result.weights,
                normalized_weights={k: round(v, 8) for k, v in result.normalized_weights.items()},
                trust_score=round(result.trust_score, 6),
                risk_level=result.risk_level.value,
                risk_flags=[f.value for f in result.risk_flags],
            )
            artifact.sha256_hash = self.generate_sha256_hash(artifact)

            span.set_attribute("evidence.score", float(artifact.trust_score))
            span.set_attribute("evidence.risk_flags_count", len(artifact.risk_flags))
            span.set_attribute("evidence.dimension_count", len(artifact.scores))

            logger.info(
                "Evidence artifact generated | id=%s | hash=%s...",
                artifact.artifact_id,
                artifact.sha256_hash[:16],
            )
            return artifact

    @staticmethod
    def generate_sha256_hash(artifact: EvidenceArtifact) -> str:
        with tracer.start_as_current_span("hash_computation") as span:
            canonical_json = artifact.to_json(include_hash=False)
            raw_bytes = canonical_json.encode("utf-8")
            digest = hashlib.sha256(raw_bytes).hexdigest()
            span.set_attribute("hash.algorithm", "sha256")
            span.set_attribute("hash.output_length", len(digest))
            span.set_attribute("hash.prefix", digest[:8])
            return digest

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)


# ---------------------------------------------------------------------------
# SAMPLE EXECUTION
# ---------------------------------------------------------------------------

def run_sample_evaluation() -> EvidenceArtifact:
    scores = {
        "accuracy":     0.88,
        "robustness":   0.72,
        "fairness":     0.51,
        "safety":       0.65,
        "privacy":      0.80,
        "transparency": 0.58,
    }

    weights = {
        "accuracy":     0.20,
        "robustness":   0.15,
        "fairness":     0.25,
        "safety":       0.25,
        "privacy":      0.10,
        "transparency": 0.05,
    }

    print("\n" + "=" * 68)
    print("  TRUST SCORE EVALUATION ENGINE — Sample Run")
    print("  Use Case: Loan Approval AI (EU AI Act High-Risk)")
    print("=" * 68)

    calc = TrustScoreCalculator(scores, weights)
    result = calc.calculate_score()

    print("\n  Dimension Scores")
    print(f"  {'Dimension':<16} {'Score':>8}  {'Weight (norm)':>14}")
    print("  " + '-' * 44)
    for dim in sorted(VALID_DIMENSIONS):
        s = result.scores.get(dim, 0.0)
        w = result.normalized_weights.get(dim, 0.0)
        print(f"  {dim:<16} {s:>8.4f}  {w:>14.4f}")

    print(f"\n  Trust Score  : {result.trust_score:.4f}")
    print(f"  Risk Level   : {result.risk_level.value}")
    print(f"  Risk Flags   : {[f.value for f in result.risk_flags] or 'None'}")

    if calc.warnings:
        print("\n  Warnings:")
        for w in calc.warnings:
            print(f"    ⚠  {w}")

    artifact = calc.last_evidence or calc.generate_evidence(result)
    if not artifact.sha256_hash:
        artifact.sha256_hash = calc.generate_sha256_hash(artifact)
    print("\n" + "=" * 68)
    print("  EVIDENCE ARTIFACT (SHA-256 Sealed)")
    print("=" * 68)
    print(artifact.to_json())

    return artifact


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------
import unittest  # noqa: E402


class TestTrustScoreEngine(unittest.TestCase):
    VALID_SCORES = {
        "accuracy": 0.85, "robustness": 0.75, "fairness": 0.70,
        "safety": 0.80,   "privacy": 0.90,    "transparency": 0.65,
    }
    VALID_WEIGHTS = {
        "accuracy": 0.20, "robustness": 0.15, "fairness": 0.20,
        "safety":   0.25, "privacy":    0.10, "transparency": 0.10,
    }

    def test_01_happy_path_returns_result(self):
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertIsInstance(result, TrustScoreResult)
        self.assertGreaterEqual(result.trust_score, 0.0)
        self.assertLessEqual(result.trust_score, 1.0)
        self.assertIsInstance(result.risk_level, RiskLevel)

    def test_02_weight_normalisation_sums_to_one(self):
        weights = {"accuracy": 2, "safety": 3, "fairness": 5}
        scores = {"accuracy": 0.8, "safety": 0.7, "fairness": 0.6}
        calc = TrustScoreCalculator(scores, weights)
        normalised = calc.normalize_weights()
        self.assertAlmostEqual(sum(normalised.values()), 1.0, places=6)

    def test_03_score_above_one_raises_validation_error(self):
        bad_scores = {**self.VALID_SCORES, "accuracy": 1.5}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_04_score_below_zero_raises_validation_error(self):
        bad_scores = {**self.VALID_SCORES, "safety": -0.1}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_05_negative_weight_raises_validation_error(self):
        bad_weights = {**self.VALID_WEIGHTS, "fairness": -0.10}
        calc = TrustScoreCalculator(self.VALID_SCORES, bad_weights)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_06_all_weights_zero_raises_normalisation_error(self):
        zero_weights = {k: 0.0 for k in self.VALID_SCORES}
        calc = TrustScoreCalculator(self.VALID_SCORES, zero_weights)
        with self.assertRaises(NormalizationError):
            calc.normalize_weights()

    def test_07_missing_dimensions_produce_warning(self):
        partial_scores = {"accuracy": 0.8, "safety": 0.7}
        partial_weights = {"accuracy": 0.5, "safety": 0.5}
        calc = TrustScoreCalculator(partial_scores, partial_weights)
        calc.validate_inputs()
        self.assertTrue(any("Missing" in w for w in calc.warnings))

    def test_08_risk_classification_critical(self):
        calc = TrustScoreCalculator({"accuracy": 0.3}, {"accuracy": 1.0})
        self.assertEqual(calc.classify_risk(0.30), RiskLevel.CRITICAL)

    def test_09_risk_classification_low(self):
        calc = TrustScoreCalculator({"accuracy": 0.9}, {"accuracy": 1.0})
        self.assertEqual(calc.classify_risk(0.90), RiskLevel.LOW)

    def test_10_risk_boundary_exactly_at_high_threshold(self):
        calc = TrustScoreCalculator({"accuracy": 0.6}, {"accuracy": 1.0})
        self.assertEqual(calc.classify_risk(0.60), RiskLevel.HIGH)

    def test_11_sha256_hash_is_64_hex_chars(self):
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        art = calc.generate_evidence(result)
        self.assertEqual(len(art.sha256_hash), 64)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", art.sha256_hash))

    def test_12_identical_inputs_produce_identical_hash(self):
        calc1 = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        calc2 = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        r1 = calc1.calculate_score()
        r2 = calc2.calculate_score()
        self.assertAlmostEqual(r1.trust_score, r2.trust_score, places=8)

    def test_13_evidence_artifact_contains_required_fields(self):
        calc = TrustScoreCalculator(self.VALID_SCORES, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        art = calc.generate_evidence(result)
        data = json.loads(art.to_json())

        required_fields = [
            "artifact_id", "timestamp", "evaluator_version",
            "scores", "weights", "normalized_weights",
            "trust_score", "risk_level", "risk_flags", "sha256_hash",
        ]
        for f in required_fields:
            self.assertIn(f, data, msg=f"Missing field: {f}")

    def test_14_all_scores_zero_produces_critical(self):
        zero_scores = {k: 0.0 for k in VALID_DIMENSIONS}
        calc = TrustScoreCalculator(zero_scores, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertEqual(result.risk_level, RiskLevel.CRITICAL)
        self.assertAlmostEqual(result.trust_score, 0.0)

    def test_15_all_scores_one_produces_low_risk(self):
        perfect_scores = {k: 1.0 for k in VALID_DIMENSIONS}
        calc = TrustScoreCalculator(perfect_scores, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        self.assertAlmostEqual(result.trust_score, 1.0, places=6)

    def test_16_empty_scores_raises_validation_error(self):
        calc = TrustScoreCalculator({}, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_17_unknown_dimension_raises_validation_error(self):
        bad_scores = {"accurasy": 0.8}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_18_risk_flag_raised_for_low_fairness(self):
        scores = {**self.VALID_SCORES, "fairness": 0.40}
        calc = TrustScoreCalculator(scores, self.VALID_WEIGHTS)
        result = calc.calculate_score()
        self.assertIn(RiskFlag.FAIRNESS_RISK, result.risk_flags)

    def test_19_extremely_large_score_raises_error(self):
        bad_scores = {**self.VALID_SCORES, "accuracy": 1e7}
        calc = TrustScoreCalculator(bad_scores, self.VALID_WEIGHTS)
        with self.assertRaises(ValidationError):
            calc.validate_inputs()

    def test_20_single_dimension_evaluates_correctly(self):
        single_score = {"accuracy": 0.72}
        single_weight = {"accuracy": 1.0}
        calc = TrustScoreCalculator(single_score, single_weight)
        result = calc.calculate_score()
        self.assertAlmostEqual(result.trust_score, 0.72, places=6)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    artifact = run_sample_evaluation()

    print("\n" + "=" * 68)
    print("  RUNNING UNIT TESTS")
    print("=" * 68 + "\n")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTrustScoreEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
