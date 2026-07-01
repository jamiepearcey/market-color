import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(new URL("../src/lib/brain/experimentArtifact.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
    verbatimModuleSyntax: false,
  },
});

const moduleUrl = `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`;
const { parseEmbeddedExperimentArtifact, parseExperimentArtifactJson, REFERENCE_EXPERIMENT_ARTIFACT } = await import(moduleUrl);

function parse(value) {
  return parseExperimentArtifactJson(JSON.stringify(value));
}

{
  const result = parse(REFERENCE_EXPERIMENT_ARTIFACT);
  assert.equal(result.ok, true);
  assert.equal(result.trials, 3);
  assert.equal(result.obs, 12);
  assert.equal(result.nPartitions, 4);
  assert.equal(result.returns.length, 36);
  assert.deepEqual(result.parameterCoordinates, [[0], [1], [2]]);
  assert.deepEqual(result.parameterNames, ["variant"]);
}

{
  const result = parse({
    n_partitions: 2,
    trials: [{ returns: [0.01, 0.02] }, { returns: [0.03, -0.01] }],
  });
  assert.equal(result.ok, true);
  assert.deepEqual(result.trialNames, ["trial 1", "trial 2"]);
}

{
  const result = parse({
    n_partitions: 2,
    parameter_coordinates: [[5, 20], [10, 20]],
    parameter_names: ["lookback", "threshold"],
    trials: [{ returns: [0.01, 0.02] }, { returns: [0.03, -0.01] }],
  });
  assert.equal(result.ok, true);
  assert.deepEqual(result.parameterCoordinates, [[5, 20], [10, 20]]);
  assert.deepEqual(result.artifact.parameter_coordinates, [[5, 20], [10, 20]]);
  assert.deepEqual(result.artifact.parameter_names, ["lookback", "threshold"]);
}

{
  const result = parse({
    n_partitions: 2,
    parameter_coordinates: [[5], [10, 20]],
    trials: [{ returns: [0.01, 0.02] }, { returns: [0.03, -0.01] }],
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /same length/);
}

{
  const result = parse({
    n_partitions: 2,
    parameter_coordinates: [[5], [10]],
    parameter_names: ["lookback", "threshold"],
    trials: [{ returns: [0.01, 0.02] }, { returns: [0.03, -0.01] }],
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /parameter_names/);
}

{
  const result = parse({
    n_partitions: 3,
    trials: [{ returns: [0.01, 0.02, 0.03] }, { returns: [0.03, -0.01, 0.02] }],
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /even n_partitions/);
}

{
  const result = parse({
    n_partitions: 2,
    trials: [{ returns: [0.01, 0.02] }, { returns: [0.03] }],
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /same number of observations/);
}

{
  const result = parse({
    n_partitions: 2,
    trials: [{ returns: [0.01, Number.NaN] }, { returns: [0.03, -0.01] }],
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /finite numbers/);
}

{
  const result = parse({
    n_partitions: 4,
    trials: [{ returns: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] }, { returns: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] }],
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /divide evenly/);
}

{
  const result = parseEmbeddedExperimentArtifact({
    id: "H-DEMO",
    validation_artifact: REFERENCE_EXPERIMENT_ARTIFACT,
  });
  assert.equal(result.present, true);
  assert.equal(result.key, "validation_artifact");
  assert.equal(result.result.ok, true);
  assert.equal(result.result.trials, 3);
}

{
  const result = parseEmbeddedExperimentArtifact({
    id: "H-DEMO",
    experiment_artifact: JSON.stringify(REFERENCE_EXPERIMENT_ARTIFACT),
  });
  assert.equal(result.present, true);
  assert.equal(result.key, "experiment_artifact");
  assert.equal(result.result.ok, true);
  assert.equal(result.result.obs, 12);
}

{
  const result = parseEmbeddedExperimentArtifact({
    id: "H-DEMO",
    validation_artifact: undefined,
  });
  assert.equal(result.present, false);
}

console.log("experiment artifact parser checks passed");
