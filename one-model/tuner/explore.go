// Package main implements the configuration exploration engine
// Executes the engine_execution_plan from mm.json
package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"time"
)

// ================================================================
// CONFIGURATION EXPLORATION ENGINE
// ================================================================

// ExperimentConfig represents a full optimizer configuration
type ExperimentConfig struct {
	Name                  string
	LearningRate          float32
	MomentumBeta          float32
	NoiseScale            float32
	EnableQuantumJumps    bool
	EnableHillClimb       bool
	JumpSeverityThreshold float32
	QuantumJumpRange      float32
	NuclearResetStrength  float32
	HillClimbTriggerStuck int
	HillClimbMaxSteps     int
	HillClimbRewardScale  float32
}

// BenchmarkProblem represents a test problem
type BenchmarkProblem struct {
	Name      string
	Function  string
	Dimension int
	Landscape string
	Optimum   float64
}

// ExperimentResult stores results from a single run
type ExperimentResult struct {
	Config           ExperimentConfig
	Problem          BenchmarkProblem
	FinalObjective   float32
	BestObjective    float32
	ConvergenceSteps int
	EscapeCount      int
	StabilityStdDev  float32
	FunctionEvals    int
	QuantumJumps     int
	NuclearJumps     int
	HillClimbCount   int
	RunTime          time.Duration
}

// ConfigExplorer manages the configuration exploration process
type ConfigExplorer struct {
	Graph    *Graph
	Configs  []ExperimentConfig
	Problems []BenchmarkProblem
	Results  []ExperimentResult
}

// NewConfigExplorer creates a new configuration explorer
func NewConfigExplorer(g *Graph) *ConfigExplorer {
	return &ConfigExplorer{
		Graph:    g,
		Configs:  make([]ExperimentConfig, 0),
		Problems: make([]BenchmarkProblem, 0),
		Results:  make([]ExperimentResult, 0),
	}
}

// ================================================================
// PHASE 1: CONFIGURATION SPACE GENERATION
// ================================================================

func (ce *ConfigExplorer) Phase1_GenerateConfigs() {
	fmt.Printf("\n=== PHASE 1: Configuration Space Generation ===\n")
	
	// Extract experiment configurations from Canon nodes
	for _, canon := range ce.Graph.ByType["Canon"] {
		if canon.Fields["category"].Raw == "experiment" {
			config := ce.extractExperimentConfig(canon)
			ce.Configs = append(ce.Configs, config)
			fmt.Printf("  Loaded config: %s\n", config.Name)
		}
	}
	
	// Also add individual config_value combinations if needed
	ce.addIndividualConfigs()
	
	fmt.Printf("  Total configurations: %d\n", len(ce.Configs))
}

func (ce *ConfigExplorer) extractExperimentConfig(canon *Node) ExperimentConfig {
	config := ExperimentConfig{
		Name: canon.Fields["name"].Raw,
	}
	
	// Extract numeric fields
	if lr := canon.Fields["learning_rate"]; lr.IsNumber {
		config.LearningRate = float32(lr.Number)
	}
	if mb := canon.Fields["momentum_beta"]; mb.IsNumber {
		config.MomentumBeta = float32(mb.Number)
	}
	if ns := canon.Fields["noise_scale"]; ns.IsNumber {
		config.NoiseScale = float32(ns.Number)
	}
	if jst := canon.Fields["jump_severity_threshold"]; jst.IsNumber {
		config.JumpSeverityThreshold = float32(jst.Number)
	}
	if qjr := canon.Fields["quantum_jump_range"]; qjr.IsNumber {
		config.QuantumJumpRange = float32(qjr.Number)
	}
	if nrs := canon.Fields["nuclear_reset_strength"]; nrs.IsNumber {
		config.NuclearResetStrength = float32(nrs.Number)
	}
	if hcts := canon.Fields["hill_climb_trigger_stuck"]; hcts.IsNumber {
		config.HillClimbTriggerStuck = int(hcts.Number)
	}
	if hcms := canon.Fields["hill_climb_max_steps"]; hcms.IsNumber {
		config.HillClimbMaxSteps = int(hcms.Number)
	}
	if hcrs := canon.Fields["hill_climb_reward_scale"]; hcrs.IsNumber {
		config.HillClimbRewardScale = float32(hcrs.Number)
	}
	
	// Extract boolean fields
	config.EnableQuantumJumps = canon.Fields["enable_quantum_jumps"].Raw == "true"
	config.EnableHillClimb = canon.Fields["enable_hill_climb"].Raw == "true"
	
	return config
}

func (ce *ConfigExplorer) addIndividualConfigs() {
	// Add some grid search points from individual config_value Canons
	lrValues := []float32{}
	momentumValues := []float32{}
	noiseValues := []float32{}
	
	for _, canon := range ce.Graph.ByType["Canon"] {
		cat := canon.Fields["category"].Raw
		if cat == "config_value" {
			if lr := canon.Fields["learning_rate"]; lr.IsNumber {
				lrValues = append(lrValues, float32(lr.Number))
			}
			if mb := canon.Fields["momentum_beta"]; mb.IsNumber {
				momentumValues = append(momentumValues, float32(mb.Number))
			}
			if ns := canon.Fields["noise_scale"]; ns.IsNumber {
				noiseValues = append(noiseValues, float32(ns.Number))
			}
		}
	}
	
	// Create a few additional grid combinations
	if len(lrValues) > 0 && len(momentumValues) > 0 && len(noiseValues) > 0 {
		for _, lr := range lrValues[:minInt(2, len(lrValues))] {
			for _, mb := range momentumValues[:minInt(2, len(momentumValues))] {
				for _, ns := range noiseValues[:minInt(2, len(noiseValues))] {
					config := ExperimentConfig{
						Name:               fmt.Sprintf("grid_lr%.4f_mb%.2f_ns%.2f", lr, mb, ns),
						LearningRate:       lr,
						MomentumBeta:       mb,
						NoiseScale:         ns,
						EnableQuantumJumps: false,
						EnableHillClimb:    true,
						HillClimbTriggerStuck: 30,
						HillClimbMaxSteps:  20,
					}
					ce.Configs = append(ce.Configs, config)
				}
			}
		}
	}
}

// ================================================================
// PHASE 2: BENCHMARK PROBLEM LOADING
// ================================================================

func (ce *ConfigExplorer) Phase2_LoadProblems() {
	fmt.Printf("\n=== PHASE 2: Benchmark Problem Loading ===\n")
	
	for _, canon := range ce.Graph.ByType["Canon"] {
		if canon.Fields["category"].Raw == "benchmark" {
			problem := BenchmarkProblem{
				Name:      canon.Fields["name"].Raw,
				Function:  canon.Fields["function"].Raw,
				Landscape: canon.Fields["landscape"].Raw,
			}
			
			if dim := canon.Fields["dimension"]; dim.IsNumber {
				problem.Dimension = int(dim.Number)
			} else {
				problem.Dimension = 10
			}
			
			if opt := canon.Fields["optimum"]; opt.IsNumber {
				problem.Optimum = opt.Number
			}
			
			ce.Problems = append(ce.Problems, problem)
			fmt.Printf("  Loaded problem: %s (%s, dim=%d)\n", 
				problem.Name, problem.Landscape, problem.Dimension)
		}
	}
	
	fmt.Printf("  Total problems: %d\n", len(ce.Problems))
}

// ================================================================
// PHASE 3: EXPERIMENTAL EXECUTION
// ================================================================

func (ce *ConfigExplorer) Phase3_RunExperiments(stepsPerRun int) {
	fmt.Printf("\n=== PHASE 3: Experimental Execution ===\n")
	fmt.Printf("  Running %d configs × %d problems = %d experiments\n",
		len(ce.Configs), len(ce.Problems), len(ce.Configs)*len(ce.Problems))
	fmt.Printf("  Steps per run: %d\n\n", stepsPerRun)
	
	totalExperiments := len(ce.Configs) * len(ce.Problems)
	currentExperiment := 0
	
	for _, config := range ce.Configs {
		for _, problem := range ce.Problems {
			currentExperiment++
			fmt.Printf("  [%d/%d] Running %s on %s...\n", 
				currentExperiment, totalExperiments, config.Name, problem.Name)
			
			result := ce.runSingleExperiment(config, problem, stepsPerRun)
			ce.Results = append(ce.Results, result)
			
			fmt.Printf("    Final: %.6e (best: %.6e, steps: %d, escapes: %d)\n",
				result.FinalObjective, result.BestObjective, 
				result.ConvergenceSteps, result.EscapeCount)
		}
	}
}

func (ce *ConfigExplorer) runSingleExperiment(config ExperimentConfig, problem BenchmarkProblem, steps int) ExperimentResult {
	startTime := time.Now()
	
	// Create PBit config from experiment config
	pbitConfig := DefaultPBitConfig()
	pbitConfig.LearningRate = config.LearningRate
	pbitConfig.MomentumBeta = config.MomentumBeta
	pbitConfig.NoiseScale = config.NoiseScale
	pbitConfig.EnableQuantumJumps = config.EnableQuantumJumps
	pbitConfig.EnableHillClimb = config.EnableHillClimb
	
	if config.JumpSeverityThreshold > 0 {
		pbitConfig.JumpConsecutiveStuckThreshold = 50
	}
	if config.HillClimbTriggerStuck > 0 {
		pbitConfig.HillClimbTriggerStuck = config.HillClimbTriggerStuck
	}
	if config.HillClimbMaxSteps > 0 {
		pbitConfig.HillClimbMaxSteps = config.HillClimbMaxSteps
	}
	if config.HillClimbRewardScale > 0 {
		pbitConfig.HillClimbRewardScale = config.HillClimbRewardScale
	}
	
	// Create SAR config
	sarConfig := DefaultSARConfig()
	sarConfig.EnableJumps = config.EnableQuantumJumps
	if config.JumpSeverityThreshold > 0 {
		sarConfig.JumpSeverityThreshold = config.JumpSeverityThreshold
	}
	if config.QuantumJumpRange > 0 {
		sarConfig.QuantumJumpRange = config.QuantumJumpRange
	}
	if config.NuclearResetStrength > 0 {
		sarConfig.NuclearResetStrength = config.NuclearResetStrength
	}
	
	// Create optimizer
	memoryManager := NewSARMemoryManager(problem.Dimension, sarConfig)
	optimizer := NewPathBasedPBit(problem.Dimension, pbitConfig, memoryManager)
	
	// Get objective and gradient functions
	objectiveFn, gradientFn := ce.getProblemFunctions(problem.Function)
	
	// Run optimization
	convergenceThreshold := 1.01 * float32(problem.Optimum) // Within 1% of optimum
	convergenceSteps := steps
	escapeCount := 0
	objectives := make([]float32, 0, steps)
	
	for step := 0; step < steps; step++ {
		result := optimizer.Step(gradientFn, objectiveFn, 100)
		objectives = append(objectives, result.Objective)
		
		// Track convergence
		if convergenceSteps == steps && result.BestObjective <= convergenceThreshold {
			convergenceSteps = step
		}
		
		// Track escapes (jumps and hill-climbs that led to improvement)
		if result.IsJump {
			escapeCount++
		}
	}
	
	// Compute stability (std dev of last 100 objectives)
	stabilityStdDev := ce.computeStability(objectives)
	
	return ExperimentResult{
		Config:           config,
		Problem:          problem,
		FinalObjective:   objectives[len(objectives)-1],
		BestObjective:    optimizer.BestObjective,
		ConvergenceSteps: convergenceSteps,
		EscapeCount:      escapeCount,
		StabilityStdDev:  stabilityStdDev,
		FunctionEvals:    steps,
		QuantumJumps:     optimizer.MemoryManager.JumpCounts[0],
		NuclearJumps:     optimizer.MemoryManager.JumpCounts[1],
		HillClimbCount:   optimizer.HillClimbCount,
		RunTime:          time.Since(startTime),
	}
}

func (ce *ConfigExplorer) getProblemFunctions(funcName string) (func([]float32) float32, func([]float32) []float32) {
	switch funcName {
	case "sphere":
		return sphereObjective, sphereGradient
	case "rastrigin":
		return rastriginObjective, rastriginGradient
	case "ackley":
		return ackleyObjective, ackleyGradient
	default:
		return sphereObjective, sphereGradient
	}
}

func (ce *ConfigExplorer) computeStability(objectives []float32) float32 {
	if len(objectives) < 100 {
		return 0
	}
	
	last100 := objectives[len(objectives)-100:]
	mean := float32(0)
	for _, obj := range last100 {
		mean += obj
	}
	mean /= float32(len(last100))
	
	variance := float32(0)
	for _, obj := range last100 {
		diff := obj - mean
		variance += diff * diff
	}
	variance /= float32(len(last100))
	
	return float32(math.Sqrt(float64(variance)))
}

// ================================================================
// PHASE 4: ANALYSIS AND RANKING
// ================================================================

func (ce *ConfigExplorer) Phase4_AnalyzeResults() {
	fmt.Printf("\n=== PHASE 4: Analysis and Ranking ===\n")
	
	// Group by landscape type
	byLandscape := make(map[string][]ExperimentResult)
	for _, result := range ce.Results {
		landscape := result.Problem.Landscape
		byLandscape[landscape] = append(byLandscape[landscape], result)
	}
	
	fmt.Printf("\nResults by landscape type:\n")
	for landscape, results := range byLandscape {
		fmt.Printf("\n  %s (%d results):\n", landscape, len(results))
		
		// Sort by best objective
		sort.Slice(results, func(i, j int) bool {
			return results[i].BestObjective < results[j].BestObjective
		})
		
		// Show top 3
		for i := 0; i < minInt(3, len(results)); i++ {
			r := results[i]
			fmt.Printf("    %d. %s: %.6e (converged: %d steps, escapes: %d)\n",
				i+1, r.Config.Name, r.BestObjective, r.ConvergenceSteps, r.EscapeCount)
		}
		
		// Store best configs
		if len(results) > 0 {
			ce.storeBestConfig(landscape, results[0])
		}
	}
}

func (ce *ConfigExplorer) storeBestConfig(landscape string, result ExperimentResult) {
	// Create a Memory node for this best config
	memory := &Node{
		CompType: "Memory",
		ID:       fmt.Sprintf("best_config_%s", landscape),
		Fields:   make(map[string]FieldValue),
	}
	
	memory.Fields["memory_id"] = FieldValue{Raw: memory.ID}
	memory.Fields["type"] = FieldValue{Raw: "semantic"}
	memory.Fields["config_name"] = FieldValue{Raw: result.Config.Name}
	memory.Fields["best_objective"] = FieldValue{
		Raw:      fmt.Sprintf("%.6e", result.BestObjective),
		Number:   float64(result.BestObjective),
		IsNumber: true,
	}
	memory.Fields["landscape"] = FieldValue{Raw: landscape}
	
	ce.Graph.addNode(memory)
}

// ================================================================
// PHASE 5: VISUALIZATION AND EXPORT
// ================================================================

func (ce *ConfigExplorer) Phase5_ExportResults(filename string) {
	fmt.Printf("\n=== PHASE 5: Visualization and Export ===\n")
	
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("  Error creating export file: %v\n", err)
		return
	}
	defer file.Close()
	
	// Write CSV header
	fmt.Fprintf(file, "Config,Problem,Landscape,FinalObjective,BestObjective,ConvergenceSteps,EscapeCount,StabilityStdDev,QuantumJumps,NuclearJumps,HillClimbCount,RunTime\n")
	
	// Write results
	for _, r := range ce.Results {
		fmt.Fprintf(file, "%s,%s,%s,%.6e,%.6e,%d,%d,%.6e,%d,%d,%d,%v\n",
			r.Config.Name,
			r.Problem.Name,
			r.Problem.Landscape,
			r.FinalObjective,
			r.BestObjective,
			r.ConvergenceSteps,
			r.EscapeCount,
			r.StabilityStdDev,
			r.QuantumJumps,
			r.NuclearJumps,
			r.HillClimbCount,
			r.RunTime,
		)
	}
	
	fmt.Printf("  Exported results to %s\n", filename)
	
	// Generate summary statistics
	ce.printSummaryStats()
}

func (ce *ConfigExplorer) printSummaryStats() {
	fmt.Printf("\n  Summary Statistics:\n")
	
	if len(ce.Results) == 0 {
		fmt.Printf("    No results to summarize\n")
		return
	}
	
	// Best overall result
	bestResult := ce.Results[0]
	for _, r := range ce.Results {
		if r.BestObjective < bestResult.BestObjective {
			bestResult = r
		}
	}
	
	fmt.Printf("    Best overall: %s on %s (%.6e)\n",
		bestResult.Config.Name, bestResult.Problem.Name, bestResult.BestObjective)
	
	// Average convergence by landscape
	byLandscape := make(map[string][]int)
	for _, r := range ce.Results {
		byLandscape[r.Problem.Landscape] = append(
			byLandscape[r.Problem.Landscape], r.ConvergenceSteps)
	}
	
	fmt.Printf("\n    Average convergence steps by landscape:\n")
	for landscape, steps := range byLandscape {
		avg := 0
		for _, s := range steps {
			avg += s
		}
		avg /= len(steps)
		fmt.Printf("      %s: %d steps\n", landscape, avg)
	}
}

// ================================================================
// MAIN EXECUTION ENTRY POINT
// ================================================================

// RunConfigurationExploration is the main entry point for the exploration engine
func RunConfigurationExploration(graphFile string, stepsPerRun int) {
	fmt.Printf("=== Configuration Exploration Engine ===\n")
	fmt.Printf("Executing engine_execution_plan from %s\n", graphFile)
	
	// Load graph
	file, err := os.Open(graphFile)
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		return
	}
	defer file.Close()
	
	graph, err := ParseJSONData(file)
	if err != nil {
		fmt.Printf("Error parsing JSON: %v\n", err)
		return
	}
	
	graph.BuildTree()
	
	// Create explorer
	explorer := NewConfigExplorer(graph)
	
	// Execute phases
	explorer.Phase1_GenerateConfigs()
	explorer.Phase2_LoadProblems()
	explorer.Phase3_RunExperiments(stepsPerRun)
	explorer.Phase4_AnalyzeResults()
	explorer.Phase5_ExportResults("config_exploration_results.csv")
	
	fmt.Printf("\n=== Exploration Complete ===\n")
}

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

// minInt returns the minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
