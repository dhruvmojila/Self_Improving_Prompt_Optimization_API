import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

// Status badge component
const StatusBadge = ({ status }) => {
  const colors = {
    pending: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    running: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    completed: "bg-green-500/20 text-green-400 border-green-500/30",
    failed: "bg-red-500/20 text-red-400 border-red-500/30",
  };
  return (
    <span
      className={`px-3 py-1 rounded-full text-sm border ${
        colors[status] || colors.pending
      }`}
    >
      {status}
    </span>
  );
};

// Progress bar component
const ProgressBar = ({ progress, message }) => (
  <div className="w-full">
    <div className="flex justify-between text-sm mb-1">
      <span className="text-gray-400">{message || "Initializing..."}</span>
      <span className="text-gray-400">{Math.round(progress)}%</span>
    </div>
    <div className="w-full bg-gray-800 rounded-full h-2">
      <div
        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
        style={{ width: `${progress}%` }}
      />
    </div>
  </div>
);

// Result card component
const ResultCard = ({ result }) => {
  if (!result) return null;

  const improvement = result.improvement_pct || 0;
  const isImproved = improvement > 0;

  return (
    <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6 mt-6">
      <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
        <span className="text-2xl">📊</span> Optimization Results
      </h3>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-gray-400">
            {(result.baseline_score * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-500 mt-1">Baseline Score</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 text-center">
          <div
            className={`text-3xl font-bold ${
              isImproved ? "text-green-400" : "text-blue-400"
            }`}
          >
            {(result.optimized_score * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-500 mt-1">Optimized Score</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 text-center">
          <div
            className={`text-3xl font-bold ${
              isImproved ? "text-green-400" : "text-gray-400"
            }`}
          >
            {isImproved ? "+" : ""}
            {improvement.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-500 mt-1">Improvement</div>
        </div>
      </div>
      {result.artifact_path && (
        <div className="mt-4 text-sm text-gray-500">
          Artifact saved:{" "}
          <code className="text-purple-400">{result.artifact_path}</code>
        </div>
      )}
    </div>
  );
};

function App() {
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [prompt, setPrompt] = useState(null);
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Check API health
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data);
      return data.status === "healthy";
    } catch (e) {
      setError("Cannot connect to API. Is the server running?");
      return false;
    }
  };

  // Upload sample dataset
  const uploadDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const csvData = `text,sentiment
I absolutely love this product! Best purchase ever!,positive
This is terrible quality. Complete waste of money.,negative
It's okay I guess. Nothing special about it.,neutral
Amazing experience! Highly recommend to everyone!,positive
Broke after one day. Very disappointed.,negative
Decent for the price. Gets the job done.,neutral`;

      const formData = new FormData();
      formData.append(
        "file",
        new Blob([csvData], { type: "text/csv" }),
        "sentiment_data.csv"
      );
      formData.append("name", "Demo Sentiment Dataset");
      formData.append("input_fields", '["text"]');
      formData.append("output_fields", '["sentiment"]');

      const res = await fetch(`${API_BASE}/api/v1/datasets/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to upload dataset");
      const data = await res.json();
      setDataset(data);
      setStep(2);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  // Create baseline prompt
  const createPrompt = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/prompts/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "Sentiment Classifier v1",
          template: "Classify the sentiment of: {text}",
          signature: { inputs: ["text"], outputs: ["sentiment"] },
          description: "Basic sentiment classification prompt",
          tags: ["demo", "sentiment"],
        }),
      });

      if (!res.ok) throw new Error("Failed to create prompt");
      const data = await res.json();
      setPrompt(data);
      setStep(3);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  // Start optimization
  const startOptimization = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/optimization/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_id: prompt.id,
          dataset_id: dataset.id,
          config: {
            teacher_model: "llama-3.3-70b-versatile",
            student_model: "llama-3.1-8b-instant",
            teacher_provider: "groq",
            student_provider: "groq",
            optimizer_type: "bootstrap",
            num_trials: 3,
            num_fewshot_examples: 3,
            budget_usd: 0.5,
            metric_name: "correctness_metric",
          },
        }),
      });

      if (!res.ok) throw new Error("Failed to start optimization");
      const data = await res.json();
      setJob(data);
      setStep(4);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  // Poll job status
  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/optimization/jobs/${job.id}`
        );
        const data = await res.json();
        setJob(data);

        if (data.status === "completed" && data.result) {
          setResult(data.result);
          setStep(5);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [job]);

  // Initialize
  useEffect(() => {
    checkHealth().then((ok) => ok && setStep(1));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 text-white p-8">
      {/* Header */}
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-12">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Prompt Optimizer
            </h1>
            <p className="text-gray-400 mt-2">
              Self-Improving AI Prompts with CI/CD
            </p>
          </div>
          {health && (
            <div className="flex items-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500/30 rounded-lg">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-green-400 text-sm">API Connected</span>
            </div>
          )}
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
            ⚠️ {error}
          </div>
        )}

        {/* Steps */}
        <div className="space-y-6">
          {/* Step 1: Dataset */}
          <div
            className={`p-6 rounded-xl border transition-all ${
              step >= 1
                ? "bg-gray-800/50 border-gray-700"
                : "bg-gray-900/30 border-gray-800 opacity-50"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${
                    step > 1
                      ? "bg-green-500"
                      : step === 1
                      ? "bg-blue-500"
                      : "bg-gray-700"
                  }`}
                >
                  {step > 1 ? "✓" : "1"}
                </div>
                <div>
                  <h2 className="text-xl font-semibold">Upload Dataset</h2>
                  <p className="text-gray-400 text-sm">
                    Upload training data for optimization
                  </p>
                </div>
              </div>
              {step === 1 && (
                <button
                  onClick={uploadDataset}
                  disabled={loading}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium transition-colors disabled:opacity-50"
                >
                  {loading ? "Uploading..." : "Upload Sample Data"}
                </button>
              )}
              {dataset && <StatusBadge status="completed" />}
            </div>
            {dataset && (
              <div className="mt-4 pl-14 text-sm text-gray-400">
                ✓ Uploaded <span className="text-white">{dataset.name}</span>{" "}
                with {dataset.total_rows} rows
              </div>
            )}
          </div>

          {/* Step 2: Prompt */}
          <div
            className={`p-6 rounded-xl border transition-all ${
              step >= 2
                ? "bg-gray-800/50 border-gray-700"
                : "bg-gray-900/30 border-gray-800 opacity-50"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${
                    step > 2
                      ? "bg-green-500"
                      : step === 2
                      ? "bg-blue-500"
                      : "bg-gray-700"
                  }`}
                >
                  {step > 2 ? "✓" : "2"}
                </div>
                <div>
                  <h2 className="text-xl font-semibold">
                    Create Baseline Prompt
                  </h2>
                  <p className="text-gray-400 text-sm">
                    Define the prompt to optimize
                  </p>
                </div>
              </div>
              {step === 2 && (
                <button
                  onClick={createPrompt}
                  disabled={loading}
                  className="px-6 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg font-medium transition-colors disabled:opacity-50"
                >
                  {loading ? "Creating..." : "Create Prompt"}
                </button>
              )}
              {prompt && <StatusBadge status="completed" />}
            </div>
            {prompt && (
              <div className="mt-4 pl-14">
                <code className="block p-3 bg-gray-900 rounded-lg text-sm text-purple-300 font-mono">
                  {prompt.template}
                </code>
              </div>
            )}
          </div>

          {/* Step 3: Optimize */}
          <div
            className={`p-6 rounded-xl border transition-all ${
              step >= 3
                ? "bg-gray-800/50 border-gray-700"
                : "bg-gray-900/30 border-gray-800 opacity-50"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${
                    step > 4
                      ? "bg-green-500"
                      : step >= 3
                      ? "bg-blue-500"
                      : "bg-gray-700"
                  }`}
                >
                  {step > 4 ? "✓" : "3"}
                </div>
                <div>
                  <h2 className="text-xl font-semibold">Run Optimization</h2>
                  <p className="text-gray-400 text-sm">
                    Bootstrap few-shot optimization with teacher-student pattern
                  </p>
                </div>
              </div>
              {step === 3 && (
                <button
                  onClick={startOptimization}
                  disabled={loading}
                  className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 rounded-lg font-medium transition-colors disabled:opacity-50"
                >
                  {loading ? "Starting..." : "🚀 Start Optimization"}
                </button>
              )}
              {job && <StatusBadge status={job.status} />}
            </div>

            {/* Progress */}
            {job && job.status === "running" && job.progress && (
              <div className="mt-4 pl-14">
                <ProgressBar
                  progress={job.progress.progress_pct || 0}
                  message={job.progress.message}
                />
              </div>
            )}

            {/* Model info */}
            {job && (
              <div className="mt-4 pl-14 flex gap-4 text-sm">
                <div className="px-3 py-1 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400">
                  👨‍🏫 Teacher: Llama-3.3-70B
                </div>
                <div className="px-3 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg text-purple-400">
                  👨‍🎓 Student: Llama-3.1-8B
                </div>
              </div>
            )}
          </div>

          {/* Results */}
          {result && <ResultCard result={result} />}

          {/* Success message */}
          {step === 5 && (
            <div className="p-6 bg-gradient-to-r from-green-500/10 to-blue-500/10 border border-green-500/30 rounded-xl text-center">
              <div className="text-4xl mb-4">🎉</div>
              <h3 className="text-2xl font-bold text-white mb-2">
                Optimization Complete!
              </h3>
              <p className="text-gray-400">
                Your prompt has been automatically optimized using
                teacher-student distillation.
                <br />
                The optimized version is ready for production deployment.
              </p>
              <button
                onClick={() => {
                  setStep(1);
                  setDataset(null);
                  setPrompt(null);
                  setJob(null);
                  setResult(null);
                }}
                className="mt-6 px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
              >
                Run Another Optimization
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          Powered by DSPy • Pixeltable • Groq • FastAPI
        </div>
      </div>
    </div>
  );
}

export default App;
