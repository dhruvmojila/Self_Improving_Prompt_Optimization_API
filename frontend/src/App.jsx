import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

// Example datasets for quick demo
const EXAMPLE_DATASETS = {
  sentiment: {
    name: "Sentiment Analysis",
    description: "Classify text as positive, negative, or neutral",
    inputField: "text",
    outputField: "sentiment",
    data: `text,sentiment
I absolutely love this product! Best purchase ever!,positive
This is terrible quality. Complete waste of money.,negative
It's okay I guess. Nothing special about it.,neutral
Amazing experience! Highly recommend to everyone!,positive
Broke after one day. Very disappointed.,negative
Decent for the price. Gets the job done.,neutral
The customer service was incredible and helpful!,positive
Never buying from this company again.,negative`,
  },
  category: {
    name: "Topic Classification",
    description: "Classify news into categories",
    inputField: "headline",
    outputField: "category",
    data: `headline,category
Stock market hits all-time high today,business
Scientists discover new exoplanet in habitable zone,science
Local team wins championship after overtime thriller,sports
New smartphone launches with revolutionary camera,technology
Climate summit reaches historic agreement,politics`,
  },
  intent: {
    name: "Intent Detection",
    description: "Detect user intent for chatbot",
    inputField: "message",
    outputField: "intent",
    data: `message,intent
What's the weather like today?,weather_query
Book me a flight to New York,booking
I want to cancel my subscription,cancellation
Tell me a joke,entertainment
What are your business hours?,information`,
  },
};

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
    <div className="w-full bg-gray-800 rounded-full h-3">
      <div
        className="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
        style={{ width: `${progress}%` }}
      />
    </div>
  </div>
);

function App() {
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  // Dataset state
  const [selectedExample, setSelectedExample] = useState("sentiment");
  const [customData, setCustomData] = useState("");
  const [datasetName, setDatasetName] = useState("My Dataset");
  const [inputField, setInputField] = useState("text");
  const [outputField, setOutputField] = useState("sentiment");
  const [dataset, setDataset] = useState(null);

  // Prompt state
  const [promptTemplate, setPromptTemplate] = useState("");
  const [promptDescription, setPromptDescription] = useState("");
  const [prompt, setPrompt] = useState(null);

  // Job state
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [optimizedPrompt, setOptimizedPrompt] = useState(null);

  // Load example dataset
  useEffect(() => {
    const example = EXAMPLE_DATASETS[selectedExample];
    if (example) {
      setCustomData(example.data);
      setDatasetName(example.name);
      setInputField(example.inputField);
      setOutputField(example.outputField);
      setPromptTemplate(
        `Classify the ${example.outputField} of: {${example.inputField}}`
      );
      setPromptDescription(example.description);
    }
  }, [selectedExample]);

  // Check API health
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data);
      return data.status === "healthy";
    } catch (e) {
      setError("Cannot connect to API. Is the server running on port 8000?");
      return false;
    }
  };

  // Upload dataset
  const uploadDataset = async () => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append(
        "file",
        new Blob([customData], { type: "text/csv" }),
        "data.csv"
      );
      formData.append("name", datasetName);
      formData.append("input_fields", JSON.stringify([inputField]));
      formData.append("output_fields", JSON.stringify([outputField]));

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

  // Create prompt
  const createPrompt = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/prompts/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `${datasetName} Classifier`,
          template: promptTemplate,
          signature: { inputs: [inputField], outputs: [outputField] },
          description: promptDescription,
          tags: ["demo"],
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
    setOptimizedPrompt(null);
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

        console.log("Job status:", data);

        if (data.status === "completed" && data.result) {
          setResult(data.result);

          // Fetch the REAL optimized prompt from artifact endpoint
          try {
            const artifactRes = await fetch(
              `${API_BASE}/api/v1/optimization/jobs/${job.id}/artifact`
            );
            if (artifactRes.ok) {
              const artifactData = await artifactRes.json();
              console.log("Artifact data:", artifactData);

              setOptimizedPrompt({
                originalTemplate: promptTemplate,
                optimizedTemplate: artifactData.optimized_prompt,
                demos: artifactData.demos,
                signature: artifactData.signature,
                artifactPath: artifactData.artifact_path,
              });
            } else {
              // Fallback to static display
              setOptimizedPrompt({
                originalTemplate: promptTemplate,
                optimizedTemplate: `Could not load artifact. Path: ${data.result.artifact_path}`,
                artifactPath: data.result.artifact_path,
              });
            }
          } catch (artifactError) {
            console.error("Error fetching artifact:", artifactError);
            setOptimizedPrompt({
              originalTemplate: promptTemplate,
              optimizedTemplate: `Optimization complete! Artifact saved at: ${data.result.artifact_path}`,
              artifactPath: data.result.artifact_path,
            });
          }

          setStep(5);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [job, promptTemplate]);

  // Load optimized prompt from artifact
  const loadOptimizedPrompt = async (artifactPath) => {
    try {
      // For demo, construct a simple optimized prompt display
      const optimized = {
        originalTemplate: promptTemplate,
        optimizedTemplate: `[OPTIMIZED] ${promptTemplate}

### Few-Shot Examples Learned:
1. "${inputField}: I love it! → ${outputField}: positive"
2. "${inputField}: Terrible! → ${outputField}: negative"
3. "${inputField}: It's okay → ${outputField}: neutral"

### Reasoning Strategy: Chain-of-Thought
The optimized prompt includes high-quality demonstrations selected by the 70B teacher model.`,
        artifactPath,
      };
      setOptimizedPrompt(optimized);
    } catch (e) {
      console.error("Error loading optimized prompt:", e);
    }
  };

  // Initialize
  useEffect(() => {
    checkHealth().then((ok) => ok && setStep(1));
  }, []);

  // Reset
  const resetFlow = () => {
    setStep(1);
    setDataset(null);
    setPrompt(null);
    setJob(null);
    setResult(null);
    setOptimizedPrompt(null);
    setSelectedExample("sentiment");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-lg sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-xl">
              🚀
            </div>
            <div>
              <h1 className="text-xl font-bold">Prompt Optimizer</h1>
              <p className="text-xs text-gray-500">CI/CD for AI Prompts</p>
            </div>
          </div>
          {health && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/30 rounded-lg">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-green-400 text-sm">Connected</span>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 flex items-center gap-2">
            <span>⚠️</span> {error}
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              ✕
            </button>
          </div>
        )}

        {/* Step 1: Dataset */}
        <section
          className={`mb-8 p-6 rounded-2xl border transition-all ${
            step >= 1
              ? "bg-gray-800/40 border-gray-700"
              : "bg-gray-900/30 border-gray-800 opacity-50"
          }`}
        >
          <div className="flex items-center gap-4 mb-6">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold ${
                step > 1
                  ? "bg-green-500"
                  : step === 1
                  ? "bg-blue-500"
                  : "bg-gray-700"
              }`}
            >
              {step > 1 ? "✓" : "1"}
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-semibold">Upload Your Dataset</h2>
              <p className="text-gray-400">
                CSV format with input and output columns
              </p>
            </div>
            {dataset && <StatusBadge status="completed" />}
          </div>

          {step === 1 && (
            <div className="space-y-6">
              {/* Example selector */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Quick Start: Choose an Example
                </label>
                <div className="flex gap-3">
                  {Object.entries(EXAMPLE_DATASETS).map(([key, ex]) => (
                    <button
                      key={key}
                      onClick={() => setSelectedExample(key)}
                      className={`px-4 py-2 rounded-lg border transition-all ${
                        selectedExample === key
                          ? "bg-blue-500/20 border-blue-500 text-blue-300"
                          : "bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600"
                      }`}
                    >
                      {ex.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Dataset name & fields */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Dataset Name
                  </label>
                  <input
                    type="text"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Input Column
                  </label>
                  <input
                    type="text"
                    value={inputField}
                    onChange={(e) => setInputField(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Output Column
                  </label>
                  <input
                    type="text"
                    value={outputField}
                    onChange={(e) => setOutputField(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </div>

              {/* CSV data editor */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  CSV Data (or paste your own)
                </label>
                <textarea
                  value={customData}
                  onChange={(e) => setCustomData(e.target.value)}
                  rows={8}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg font-mono text-sm focus:border-blue-500 focus:outline-none"
                  placeholder="Enter CSV data..."
                />
              </div>

              <button
                onClick={uploadDataset}
                disabled={loading || !customData.trim()}
                className="w-full py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Uploading..." : "📤 Upload Dataset"}
              </button>
            </div>
          )}

          {dataset && step > 1 && (
            <div className="bg-gray-900/50 rounded-lg p-4 text-sm">
              ✓ <span className="text-white font-medium">{dataset.name}</span>{" "}
              uploaded with{" "}
              <span className="text-blue-400">{dataset.total_rows} rows</span>
            </div>
          )}
        </section>

        {/* Step 2: Prompt */}
        <section
          className={`mb-8 p-6 rounded-2xl border transition-all ${
            step >= 2
              ? "bg-gray-800/40 border-gray-700"
              : "bg-gray-900/30 border-gray-800 opacity-50"
          }`}
        >
          <div className="flex items-center gap-4 mb-6">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold ${
                step > 2
                  ? "bg-green-500"
                  : step === 2
                  ? "bg-purple-500"
                  : "bg-gray-700"
              }`}
            >
              {step > 2 ? "✓" : "2"}
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-semibold">Write Your Prompt</h2>
              <p className="text-gray-400">
                Use {"{"}variable{"}"} syntax for dynamic inputs
              </p>
            </div>
            {prompt && <StatusBadge status="completed" />}
          </div>

          {step === 2 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Prompt Template
                </label>
                <textarea
                  value={promptTemplate}
                  onChange={(e) => setPromptTemplate(e.target.value)}
                  rows={3}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg font-mono focus:border-purple-500 focus:outline-none"
                  placeholder={`e.g., Classify the sentiment of: {${inputField}}`}
                />
                <p className="mt-2 text-xs text-gray-500">
                  Tip: Use{" "}
                  <code className="text-purple-400">{`{${inputField}}`}</code>{" "}
                  to insert the input data
                </p>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Description (optional)
                </label>
                <input
                  type="text"
                  value={promptDescription}
                  onChange={(e) => setPromptDescription(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg focus:border-purple-500 focus:outline-none"
                  placeholder="What does this prompt do?"
                />
              </div>

              <button
                onClick={createPrompt}
                disabled={loading || !promptTemplate.trim()}
                className="w-full py-3 bg-purple-600 hover:bg-purple-500 rounded-xl font-semibold transition-colors disabled:opacity-50"
              >
                {loading ? "Creating..." : "✨ Create Baseline Prompt"}
              </button>
            </div>
          )}

          {prompt && step > 2 && (
            <div className="bg-gray-900/50 rounded-lg p-4">
              <code className="text-purple-300 font-mono">
                {prompt.template}
              </code>
            </div>
          )}
        </section>

        {/* Step 3: Optimize */}
        <section
          className={`mb-8 p-6 rounded-2xl border transition-all ${
            step >= 3
              ? "bg-gray-800/40 border-gray-700"
              : "bg-gray-900/30 border-gray-800 opacity-50"
          }`}
        >
          <div className="flex items-center gap-4 mb-6">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold ${
                step >= 5
                  ? "bg-green-500"
                  : step >= 3
                  ? "bg-gradient-to-br from-blue-500 to-purple-500"
                  : "bg-gray-700"
              }`}
            >
              {step >= 5 ? "✓" : "3"}
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-semibold">Run Optimization</h2>
              <p className="text-gray-400">
                AI automatically finds the best prompt version
              </p>
            </div>
            {job && <StatusBadge status={job.status} />}
          </div>

          {step === 3 && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                  <div className="flex items-center gap-2 text-blue-400 mb-2">
                    <span className="text-xl">👨‍🏫</span>
                    <span className="font-medium">Teacher Model</span>
                  </div>
                  <div className="text-gray-300">Llama-3.3-70B</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Generates high-quality examples
                  </div>
                </div>
                <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                  <div className="flex items-center gap-2 text-purple-400 mb-2">
                    <span className="text-xl">👨‍🎓</span>
                    <span className="font-medium">Student Model</span>
                  </div>
                  <div className="text-gray-300">Llama-3.1-8B</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Production-optimized inference
                  </div>
                </div>
              </div>

              <button
                onClick={startOptimization}
                disabled={loading}
                className="w-full py-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-500 hover:via-purple-500 hover:to-pink-500 rounded-xl font-bold text-lg transition-all disabled:opacity-50"
              >
                {loading ? "Starting..." : "🚀 Start Optimization"}
              </button>
            </div>
          )}

          {/* Progress */}
          {job && (job.status === "running" || job.status === "pending") && (
            <div className="mt-6">
              <ProgressBar
                progress={job.progress?.progress_pct || 0}
                message={job.progress?.message || "Waiting to start..."}
              />
            </div>
          )}
        </section>

        {/* Results */}
        {result && optimizedPrompt && (
          <section className="mb-8 p-6 rounded-2xl bg-gradient-to-br from-green-500/10 via-blue-500/10 to-purple-500/10 border border-green-500/30">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center text-2xl">
                🎉
              </div>
              <div>
                <h2 className="text-2xl font-bold">Optimization Complete!</h2>
                <p className="text-gray-400">
                  Your prompt has been automatically improved
                </p>
              </div>
            </div>

            {/* Scores */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-900/70 rounded-xl p-4 text-center">
                <div className="text-3xl font-bold text-gray-400">
                  {(result.baseline_score * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-500 mt-1">Baseline Score</div>
              </div>
              <div className="bg-gray-900/70 rounded-xl p-4 text-center">
                <div className="text-3xl font-bold text-green-400">
                  {(result.optimized_score * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-500 mt-1">
                  Optimized Score
                </div>
              </div>
              <div className="bg-gray-900/70 rounded-xl p-4 text-center">
                <div
                  className={`text-3xl font-bold ${
                    result.improvement_pct >= 0
                      ? "text-green-400"
                      : "text-gray-400"
                  }`}
                >
                  {result.improvement_pct >= 0 ? "+" : ""}
                  {result.improvement_pct.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-500 mt-1">Improvement</div>
              </div>
            </div>

            {/* Optimized Prompt Display */}
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>📝</span> Your Optimized Prompt
              </h3>

              {/* Instructions */}
              {optimizedPrompt.signature && (
                <div className="mb-4">
                  <div className="text-xs text-blue-400 mb-1">
                    🎯 OPTIMIZED INSTRUCTIONS
                  </div>
                  <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-300 font-medium">
                    {optimizedPrompt.signature.instructions ||
                      "Given the input fields, produce the output fields."}
                  </div>
                </div>
              )}

              {/* Original vs Optimized */}
              <div className="space-y-4 mb-4">
                <div>
                  <div className="text-xs text-gray-500 mb-1">
                    ORIGINAL TEMPLATE
                  </div>
                  <code className="block p-3 bg-gray-800 rounded-lg text-gray-400 font-mono text-sm line-through">
                    {optimizedPrompt.originalTemplate}
                  </code>
                </div>
              </div>

              {/* Learned Few-Shot Examples */}
              {optimizedPrompt.demos && optimizedPrompt.demos.length > 0 && (
                <div className="mt-6">
                  <div className="text-xs text-green-400 mb-3">
                    🎓 LEARNED FEW-SHOT EXAMPLES ({optimizedPrompt.demos.length}{" "}
                    demos with Chain-of-Thought reasoning)
                  </div>
                  <div className="space-y-3">
                    {optimizedPrompt.demos.map((demo, idx) => (
                      <div
                        key={idx}
                        className="p-4 bg-gradient-to-br from-green-500/10 to-blue-500/10 border border-green-500/30 rounded-lg"
                      >
                        <div className="text-xs text-gray-500 mb-2">
                          Example {idx + 1}
                        </div>
                        {Object.entries(demo).map(([key, value]) => {
                          if (key === "augmented") return null;
                          const keyDisplay =
                            key.charAt(0).toUpperCase() + key.slice(1);
                          const isReasoning = key === "reasoning";
                          return (
                            <div key={key} className="mb-2">
                              <span
                                className={`text-xs font-medium ${
                                  isReasoning
                                    ? "text-purple-400"
                                    : "text-gray-400"
                                }`}
                              >
                                {keyDisplay}:
                              </span>
                              <div
                                className={`mt-1 text-sm ${
                                  isReasoning
                                    ? "text-purple-300 italic"
                                    : "text-green-300"
                                }`}
                              >
                                {isReasoning ? `"${value}"` : value}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Fallback if no demos */}
              {(!optimizedPrompt.demos || optimizedPrompt.demos.length === 0) &&
                optimizedPrompt.optimizedTemplate && (
                  <div>
                    <div className="text-xs text-green-400 mb-1">
                      ✨ OPTIMIZED OUTPUT
                    </div>
                    <pre className="p-4 bg-gradient-to-br from-green-500/10 to-blue-500/10 border border-green-500/30 rounded-lg text-green-300 font-mono text-sm whitespace-pre-wrap">
                      {optimizedPrompt.optimizedTemplate}
                    </pre>
                  </div>
                )}

              <div className="mt-4 text-xs text-gray-500">
                Artifact saved to:{" "}
                <code className="text-purple-400">{result.artifact_path}</code>
              </div>
            </div>

            <button
              onClick={resetFlow}
              className="mt-6 w-full py-3 bg-gray-700 hover:bg-gray-600 rounded-xl font-medium transition-colors"
            >
              🔄 Optimize Another Prompt
            </button>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-6 mt-12">
        <div className="max-w-6xl mx-auto px-6 text-center text-gray-600 text-sm">
          Powered by <span className="text-blue-400">DSPy</span> •{" "}
          <span className="text-purple-400">Pixeltable</span> •{" "}
          <span className="text-green-400">Groq</span> •{" "}
          <span className="text-pink-400">FastAPI</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
