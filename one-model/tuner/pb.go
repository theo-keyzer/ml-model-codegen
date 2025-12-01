// Package pbit implements Path-Based PBit optimizer with SAR Memory Manager
// Translated from Python reference implementation v2.3
//package pbit
package main

import (
	"math"
	"math/rand"
)

// ================================================================
// CONFIGURATION STRUCTURES
// ================================================================

// SARConfig configures the SAR Memory Manager
type SARConfig struct {
	SPFDepth                 int     // Size of stuck-points FIFO
	AvoidanceThreshold       float32 // Distance threshold for avoidance
	AvoidanceStrength        float32 // Push strength away from stuck points
	EffectivenessDecay       float32 // Decay factor for strategy effectiveness
	QuantumJumpRange         float32 // Range for quantum jumps
	NuclearResetStrength     float32 // Noise scale for nuclear jumps
	MinJumpDistance          float32 // Minimum displacement for jumps
	JumpSeverityThreshold    float32 // Severity threshold for triggering jumps
	EnableJumps              bool    // Enable quantum/nuclear jumps
	Seed                     int64   // RNG seed
}

// DefaultSARConfig returns default SAR configuration
func DefaultSARConfig() SARConfig {
	return SARConfig{
		SPFDepth:                 25,
		AvoidanceThreshold:       0.3,
		AvoidanceStrength:        0.6,
		EffectivenessDecay:       0.99,
		QuantumJumpRange:         10.0,
		NuclearResetStrength:     6.0,
		MinJumpDistance:          3.0,
		JumpSeverityThreshold:    0.3,
		EnableJumps:              false,
		Seed:                     42,
	}
}

// PBitConfig configures the Path-Based PBit optimizer
type PBitConfig struct {
	NumBits                         int              // Bits per node
	NumNodes                        int              // Number of nodes (for lattices)
	CouplingStrength                float32          // Base coupling strength
	Temperature                     float32          // Temperature for sigmoid steering
	MomentumBeta                    float32          // Momentum coefficient
	MomentumDecayOnStuck            float32          // Momentum decay when stuck
	AvoidanceThreshold              float32          // Distance threshold for avoidance
	LearningRate                    float32          // Base learning rate
	NoiseScale                      float32          // Noise scaling factor
	ClipParams                      [2]float32       // Parameter clipping range
	ClipVelocity                    [2]float32       // Velocity clipping range
	ClipDelta                       [2]float32       // Delta clipping range
	Seed                            int64            // RNG seed
	Alpha                           float32          // Differential pair alpha
	Beta                            float32          // Differential pair beta
	Gamma                           float32          // Differential pair gamma
	ITail                           float32          // Tail current
	EnableQuantumJumps              bool             // Enable quantum jumps
	JumpConsecutiveStuckThreshold   int              // Stuck steps before forced jump
	PostJumpMomentumFactor          float32          // Momentum factor after jump
	EnableHillClimb                 bool             // Enable hill-climb escape
	HillClimbTriggerStuck           int              // Steps to trigger hill-climb
	HillClimbMaxSteps               int              // Max steps in hill-climb mode
	HillClimbImprovementThreshold   float32          // Relative improvement threshold
	HillClimbRewardScale            float32          // Reward scale for hill-climb
}

// DefaultPBitConfig returns default PBit configuration
func DefaultPBitConfig() PBitConfig {
	return PBitConfig{
		NumBits:                         4,
		NumNodes:                        1,
		CouplingStrength:                0.5,
		Temperature:                     1.0,
		MomentumBeta:                    0.9,
		MomentumDecayOnStuck:            0.1,
		AvoidanceThreshold:              0.3,
		LearningRate:                    0.002,
		NoiseScale:                      0.12,
		ClipParams:                      [2]float32{-2.5, 2.5},
		ClipVelocity:                    [2]float32{-0.25, 0.25},
		ClipDelta:                       [2]float32{-0.4, 0.4},
		Seed:                            42,
		Alpha:                           2.0,
		Beta:                            1.0,
		Gamma:                           0.12,
		ITail:                           5e-5,
		EnableQuantumJumps:              false,
		JumpConsecutiveStuckThreshold:   50,
		PostJumpMomentumFactor:          0.0,
		EnableHillClimb:                 true,
		HillClimbTriggerStuck:           30,
		HillClimbMaxSteps:               20,
		HillClimbImprovementThreshold:   0.05,
		HillClimbRewardScale:            1.5,
	}
}

// ================================================================
// ENUMERATIONS
// ================================================================

// ResetStrategy defines reset strategy types
type ResetStrategy int

const (
	PerturbBest ResetStrategy = iota
	BestParams
	RandomRestart
	GradientEscape
	AvoidanceRestart
	QuantumJump
	NuclearJump
)

func (rs ResetStrategy) String() string {
	names := []string{"PERTURB_BEST", "BEST_PARAMS", "RANDOM_RESTART", 
		"GRADIENT_ESCAPE", "AVOIDANCE_RESTART", "QUANTUM_JUMP", "NUCLEAR_JUMP"}
	if rs < 0 || int(rs) >= len(names) {
		return "UNKNOWN"
	}
	return names[rs]
}

// ClimbState defines hill-climb state machine states
type ClimbState int

const (
	Normal ClimbState = iota
	EscapeClimb
)

func (cs ClimbState) String() string {
	if cs == Normal {
		return "NORMAL"
	}
	return "ESCAPE_CLIMB"
}

// ================================================================
// SAR MEMORY MANAGER
// ================================================================

// SARMemoryManager manages stuck-point avoidance and reset strategies
type SARMemoryManager struct {
	ProblemDim            int
	Config                SARConfig
	StuckPointsFIFO       [][]float32 // [spf_depth][problem_dim]
	StuckPointsPtr        int
	StrategyEffectiveness []float32 // [7] strategies
	AvoidanceActive       bool
	JumpCounts            [2]int // [quantum, nuclear]
	RNG                   *rand.Rand
}

// NewSARMemoryManager creates a new SAR memory manager
func NewSARMemoryManager(problemDim int, config SARConfig) *SARMemoryManager {
	mm := &SARMemoryManager{
		ProblemDim:            problemDim,
		Config:                config,
		StuckPointsFIFO:       make([][]float32, config.SPFDepth),
		StuckPointsPtr:        0,
		StrategyEffectiveness: make([]float32, 7),
		AvoidanceActive:       false,
		RNG:                   rand.New(rand.NewSource(config.Seed)),
	}
	
	for i := range mm.StuckPointsFIFO {
		mm.StuckPointsFIFO[i] = make([]float32, problemDim)
	}
	
	return mm
}

// UpdateStuckPointsFIFO adds a stuck point to the FIFO
func (mm *SARMemoryManager) UpdateStuckPointsFIFO(stuckPoint []float32) {
	idx := mm.StuckPointsPtr % mm.Config.SPFDepth
	copy(mm.StuckPointsFIFO[idx], stuckPoint)
	mm.StuckPointsPtr = (mm.StuckPointsPtr + 1) % mm.Config.SPFDepth
}

// ComputeMinStuckDistance computes minimum distance to stuck points
func (mm *SARMemoryManager) ComputeMinStuckDistance(currentParams []float32) (float32, bool) {
	if mm.StuckPointsPtr == 0 {
		return float32(math.Inf(1)), false
	}
	
	minDist := float32(math.Inf(1))
	maxCheck := mm.StuckPointsPtr
	if maxCheck > mm.Config.SPFDepth {
		maxCheck = mm.Config.SPFDepth
	}
	
	for i := 0; i < maxCheck; i++ {
		dist := euclideanDistance(mm.StuckPointsFIFO[i], currentParams)
		if dist < minDist {
			minDist = dist
		}
	}
	
	tooClose := minDist < mm.Config.AvoidanceThreshold
	return minDist, tooClose
}

// SelectStrategy selects a reset strategy based on severity and effectiveness
func (mm *SARMemoryManager) SelectStrategy(stepsSinceImprovement, resetPatience int) (ResetStrategy, float32) {
	severity := float32(stepsSinceImprovement) / float32(max(resetPatience, 1))
	if severity > 1.0 {
		severity = 1.0
	}
	
	// Base weights for strategies
	baseWeights := []float32{0.15, 0.15, 0.25, 0.2, 0.2, 0.025, 0.025}
	
	// Adjust weights based on severity and effectiveness
	adjustedWeights := make([]float32, 7)
	totalWeight := float32(0.0)
	
	for i := 0; i < 7; i++ {
		weight := baseWeights[i]
		
		// Boost jumps if severity is high
		if severity > mm.Config.JumpSeverityThreshold && (i == int(QuantumJump) || i == int(NuclearJump)) {
			if mm.Config.EnableJumps {
				weight *= 1.1
			} else {
				weight = 0
			}
		}
		
		// Add effectiveness bonus
		weight += mm.StrategyEffectiveness[i] * 0.2
		
		adjustedWeights[i] = weight
		totalWeight += weight
	}
	
	// Normalize
	for i := range adjustedWeights {
		adjustedWeights[i] /= totalWeight
	}
	
	// Sample strategy
	r := mm.RNG.Float32()
	cumulative := float32(0.0)
	chosen := RandomRestart
	
	for i, w := range adjustedWeights {
		cumulative += w
		if r < cumulative {
			chosen = ResetStrategy(i)
			break
		}
	}
	
	// Compute strength
	strength := clip(0.3*(1.0+severity*0.7), 0.1, 1.0)
	
	return chosen, strength
}

// ResetInfo contains information about a reset operation
type ResetInfo struct {
	NewParams          []float32
	Strategy           ResetStrategy
	Strength           float32
	MinStuckDistance   float32
	AvoidanceTriggered bool
	IsJump             bool
	JumpType           string
}

// PerformReset executes a parameter reset using selected strategy
func (mm *SARMemoryManager) PerformReset(currentParams, bestParams, gradient []float32, 
	stepsSinceImprovement, resetPatience int) ResetInfo {
	
	minDistance, tooClose := mm.ComputeMinStuckDistance(currentParams)
	
	strategy, strength := mm.SelectStrategy(stepsSinceImprovement, resetPatience)
	
	// Force avoidance if too close
	if tooClose {
		strategy = AvoidanceRestart
	}
	
	// Execute reset
	newParams := make([]float32, mm.ProblemDim)
	isJump := false
	jumpType := ""
	
	switch strategy {
	case PerturbBest:
		mm.perturbBestReset(newParams, currentParams, bestParams, strength)
	case BestParams:
		mm.bestParamsReset(newParams, bestParams, strength)
	case RandomRestart:
		mm.randomRestartReset(newParams)
	case GradientEscape:
		mm.gradientEscapeReset(newParams, currentParams, gradient, strength)
	case AvoidanceRestart:
		mm.avoidanceRestartReset(newParams, currentParams, strength)
	case QuantumJump:
		if mm.Config.EnableJumps {
			mm.quantumJumpReset(newParams, currentParams, strength)
			isJump = true
			jumpType = "QUANTUM"
		} else {
			mm.randomRestartReset(newParams)
		}
	case NuclearJump:
		if mm.Config.EnableJumps {
			mm.nuclearJumpReset(newParams, currentParams)
			isJump = true
			jumpType = "NUCLEAR"
		} else {
			mm.randomRestartReset(newParams)
		}
	}
	
	// Update stuck points FIFO
	mm.UpdateStuckPointsFIFO(currentParams)
	
	// Update effectiveness
	mm.StrategyEffectiveness[int(strategy)] += 1.0
	for i := range mm.StrategyEffectiveness {
		mm.StrategyEffectiveness[i] *= mm.Config.EffectivenessDecay
	}
	
	// Update jump counts
	if isJump {
		if strategy == QuantumJump {
			mm.JumpCounts[0]++
		} else {
			mm.JumpCounts[1]++
		}
	}
	
	mm.AvoidanceActive = tooClose
	
	return ResetInfo{
		NewParams:          newParams,
		Strategy:           strategy,
		Strength:           strength,
		MinStuckDistance:   minDistance,
		AvoidanceTriggered: tooClose,
		IsJump:             isJump,
		JumpType:           jumpType,
	}
}

// ================================================================
// RESET STRATEGY IMPLEMENTATIONS
// ================================================================

func (mm *SARMemoryManager) perturbBestReset(out, current, best []float32, strength float32) {
	for i := range out {
		noise := float32(mm.RNG.NormFloat64()) * strength * 0.5
		base := current[i]*(1-strength) + best[i]*strength
		out[i] = clip(base+noise, -5.0, 5.0)
	}
}

func (mm *SARMemoryManager) bestParamsReset(out, best []float32, strength float32) {
	for i := range out {
		perturbation := float32(mm.RNG.NormFloat64()) * strength * 0.1
		out[i] = clip(best[i]+perturbation, -5.0, 5.0)
	}
}

func (mm *SARMemoryManager) randomRestartReset(out []float32) {
	for i := range out {
		out[i] = mm.RNG.Float32()*6.0 - 3.0 // [-3, 3]
	}
}

func (mm *SARMemoryManager) gradientEscapeReset(out, current, gradient []float32, strength float32) {
	gradNorm := norm(gradient)
	
	if gradNorm < 1e-10 {
		// Use random direction
		for i := range out {
			out[i] = float32(mm.RNG.NormFloat64()) * 0.1
		}
		return
	}
	
	// Escape opposite to gradient
	for i := range out {
		safeGrad := gradient[i] / (gradNorm + 1e-10)
		escapeDir := -safeGrad * strength * 2.0
		out[i] = clip(current[i]+escapeDir, -5.0, 5.0)
	}
}

func (mm *SARMemoryManager) avoidanceRestartReset(out, current []float32, strength float32) {
	if mm.StuckPointsPtr == 0 {
		mm.randomRestartReset(out)
		return
	}
	
	// Find closest stuck point
	minDist := float32(math.Inf(1))
	closestIdx := 0
	maxCheck := min(mm.StuckPointsPtr, mm.Config.SPFDepth)
	
	for i := 0; i < maxCheck; i++ {
		dist := euclideanDistance(mm.StuckPointsFIFO[i], current)
		if dist < minDist {
			minDist = dist
			closestIdx = i
		}
	}
	
	// Compute avoidance direction
	avoidanceNorm := float32(0.0)
	for i := range out {
		out[i] = current[i] - mm.StuckPointsFIFO[closestIdx][i]
		avoidanceNorm += out[i] * out[i]
	}
	avoidanceNorm = float32(math.Sqrt(float64(avoidanceNorm)))
	
	if avoidanceNorm < 1e-10 {
		// Random direction
		for i := range out {
			out[i] = float32(mm.RNG.NormFloat64())
		}
		avoidanceNorm = norm(out)
	}
	
	// Push away
	push := mm.Config.AvoidanceStrength * strength
	for i := range out {
		dir := out[i] / avoidanceNorm
		out[i] = clip(current[i]+dir*push, -5.0, 5.0)
	}
}

func (mm *SARMemoryManager) quantumJumpReset(out, current []float32, strength float32) {
	useUniform := mm.RNG.Float32() < 0.5
	
	if useUniform {
		// Uniform random in range
		for i := range out {
			out[i] = mm.RNG.Float32()*2*mm.Config.QuantumJumpRange - mm.Config.QuantumJumpRange
		}
	} else {
		// Opposite direction
		for i := range out {
			noise := float32(mm.RNG.NormFloat64()) * strength
			out[i] = -current[i] + noise
		}
	}
	
	// Ensure minimum distance
	jumpDist := euclideanDistance(out, current)
	if jumpDist < mm.Config.MinJumpDistance {
		scale := mm.Config.MinJumpDistance / (jumpDist + 1e-10)
		for i := range out {
			out[i] = current[i] + (out[i]-current[i])*scale
		}
	}
	
	// Clip
	for i := range out {
		out[i] = clip(out[i], -5.0, 5.0)
	}
}

func (mm *SARMemoryManager) nuclearJumpReset(out, current []float32) {
	// Extreme random position
	for i := range out {
		out[i] = mm.RNG.Float32()*14.0 - 7.0 // [-7, 7]
	}
	
	// Ensure minimum distance
	jumpDist := euclideanDistance(out, current)
	if jumpDist < mm.Config.MinJumpDistance {
		scale := mm.Config.MinJumpDistance / (jumpDist + 1e-10)
		for i := range out {
			out[i] = current[i] + (out[i]-current[i])*scale
		}
	}
	
	// Clip
	for i := range out {
		out[i] = clip(out[i], -10.0, 10.0)
	}
}

// ================================================================
// PATH-BASED PBIT OPTIMIZER
// ================================================================

// PathBasedPBit implements the probabilistic bit optimizer
type PathBasedPBit struct {
	ProblemDim         int
	Config             PBitConfig
	MemoryManager      *SARMemoryManager
	Params             [][]float32 // [num_nodes][num_bits]
	Velocity           [][]float32 // [num_nodes][num_bits]
	BestParams         []float32   // [problem_dim]
	BestObjective      float32
	StepCount          int
	StepsSinceImprovement int
	ConsecutiveStuck   int
	ClimbState         ClimbState
	ClimbStartObjective float32
	ClimbWorstObjective float32
	ClimbSteps         int
	HillClimbCount     int
	ObjectiveHistory   []float32
	RNG                *rand.Rand
	
	// Physical constants
	NoisePower float32
	VTKappa    float32
}

// NewPathBasedPBit creates a new PBit optimizer
func NewPathBasedPBit(problemDim int, config PBitConfig, memoryManager *SARMemoryManager) *PathBasedPBit {
	// Adjust config to match problem dimension
	configuredDim := config.NumNodes * config.NumBits
	if problemDim != configuredDim {
		if problemDim%config.NumBits == 0 {
			config.NumNodes = problemDim / config.NumBits
		} else {
			config.NumNodes = problemDim
			config.NumBits = 1
		}
	}
	
	actualDim := config.NumNodes * config.NumBits
	
	pb := &PathBasedPBit{
		ProblemDim:          actualDim,
		Config:              config,
		Params:              make([][]float32, config.NumNodes),
		Velocity:            make([][]float32, config.NumNodes),
		BestParams:          make([]float32, actualDim),
		BestObjective:       float32(math.Inf(1)),
		StepCount:           0,
		StepsSinceImprovement: 0,
		ConsecutiveStuck:    0,
		ClimbState:          Normal,
		ClimbStartObjective: float32(math.Inf(1)),
		ClimbWorstObjective: float32(math.Inf(1)),
		ClimbSteps:          0,
		HillClimbCount:      0,
		ObjectiveHistory:    make([]float32, 0, 10000),
		RNG:                 rand.New(rand.NewSource(config.Seed)),
	}
	
	// Initialize params and velocity
	for i := range pb.Params {
		pb.Params[i] = make([]float32, config.NumBits)
		pb.Velocity[i] = make([]float32, config.NumBits)
		for j := range pb.Params[i] {
			pb.Params[i][j] = float32(pb.RNG.NormFloat64()) * 0.1
		}
	}
	
	// Copy to best params (flattened)
	flattenTo(pb.Params, pb.BestParams)
	
	// Physical constants
	const (
		BOLTZMANN   = 1.3806e-23
		TEMPERATURE = 300.0
		GAMMA_NOISE = 2.0 / 3.0
		V_T         = 25.85e-3
		KAPPA       = 0.85
	)
	
	pb.NoisePower = 4 * BOLTZMANN * TEMPERATURE * GAMMA_NOISE * 100e-6 * 50e6
	pb.VTKappa = KAPPA * V_T
	
	// Create memory manager if not provided
	if memoryManager == nil {
		sarConfig := DefaultSARConfig()
		sarConfig.EnableJumps = config.EnableQuantumJumps
		pb.MemoryManager = NewSARMemoryManager(actualDim, sarConfig)
	} else {
		pb.MemoryManager = memoryManager
	}
	
	return pb
}

// StepResult contains the result of an optimization step
type StepResult struct {
	Params               []float32
	Objective            float32
	GradientNorm         float32
	BestObjective        float32
	StepsSinceImprovement int
	ConsecutiveStuck     int
	MinStuckDistance     float32
	TooCloseToStuck      bool
	NeedsReset           bool
	IsJump               bool
	JumpType             string
	ClimbState           string
	ClimbSteps           int
	InEscapeClimb        bool
	AdaptiveLR           float32
	NoiseScale           float32
}

// Step performs one optimization step
func (pb *PathBasedPBit) Step(gradientFn func([]float32) []float32, 
	objectiveFn func([]float32) float32, resetPatience int) StepResult {
	
	// Flatten params for gradient computation
	paramsFlat := make([]float32, pb.ProblemDim)
	flattenTo(pb.Params, paramsFlat)
	
	// Compute gradient and objective
	gradFlat := gradientFn(paramsFlat)
	currentObj := objectiveFn(paramsFlat)
	pb.ObjectiveHistory = append(pb.ObjectiveHistory, currentObj)
	
	// Update climb state
	pb.updateClimbState(currentObj)
	
	// Check for improvement
	hasImproved := false
	if pb.ClimbState != EscapeClimb {
		hasImproved = currentObj < pb.BestObjective-1e-8
		if hasImproved {
			pb.BestObjective = currentObj
			copy(pb.BestParams, paramsFlat)
			pb.StepsSinceImprovement = 0
			pb.ConsecutiveStuck = 0
		} else {
			pb.StepsSinceImprovement++
			pb.ConsecutiveStuck++
		}
	}
	
	// Compute gradient norm
	gradNorm := norm(gradFlat)
	
	// Reshape gradient to [num_nodes][num_bits]
	gradReshaped := make([][]float32, pb.Config.NumNodes)
	for i := range gradReshaped {
		gradReshaped[i] = make([]float32, pb.Config.NumBits)
		for j := range gradReshaped[i] {
			gradReshaped[i][j] = gradFlat[i*pb.Config.NumBits+j]
		}
	}
	
	// Compute parameter update
	deltaParams := pb.computeDelta(gradReshaped, gradNorm, currentObj)
	
	// Update velocity with momentum
	for i := range pb.Velocity {
		for j := range pb.Velocity[i] {
			pb.Velocity[i][j] = pb.Config.MomentumBeta*pb.Velocity[i][j] +
				(1-pb.Config.MomentumBeta)*deltaParams[i][j]
			pb.Velocity[i][j] = clip(pb.Velocity[i][j], 
				pb.Config.ClipVelocity[0], pb.Config.ClipVelocity[1])
		}
	}
	
	// Check stuck distance and lose momentum if too close
	minDistance, tooClose := pb.MemoryManager.ComputeMinStuckDistance(paramsFlat)
	if tooClose {
		decay := pb.Config.MomentumDecayOnStuck
		for i := range pb.Velocity {
			for j := range pb.Velocity[i] {
				pb.Velocity[i][j] *= decay
			}
		}
		pb.MemoryManager.UpdateStuckPointsFIFO(paramsFlat)
	}
	
	// Update parameters
	for i := range pb.Params {
		for j := range pb.Params[i] {
			pb.Params[i][j] = clip(pb.Params[i][j]+pb.Velocity[i][j],
				pb.Config.ClipParams[0], pb.Config.ClipParams[1])
		}
	}
	
	// Check if reset is needed
	needsReset := pb.StepsSinceImprovement > resetPatience && 
		pb.ClimbState != EscapeClimb
	forceJump := pb.Config.EnableQuantumJumps && 
		pb.ConsecutiveStuck > pb.Config.JumpConsecutiveStuckThreshold
	
	isJump := false
	jumpType := ""
	
	if needsReset || forceJump {
		effectivePatience := resetPatience
		if forceJump {
			effectivePatience = resetPatience / 2
		}
		
		resetInfo := pb.MemoryManager.PerformReset(
			paramsFlat, pb.BestParams, gradFlat, 
			pb.StepsSinceImprovement, effectivePatience)
		
		// Reshape reset params
		unflattenFrom(resetInfo.NewParams, pb.Params)
		
		// Reset velocity
		for i := range pb.Velocity {
			for j := range pb.Velocity[i] {
				pb.Velocity[i][j] = 0
			}
		}
		
		pb.ConsecutiveStuck = 0
		isJump = resetInfo.IsJump
		jumpType = resetInfo.JumpType
	}
	
	pb.StepCount++
	
	flattenTo(pb.Params, paramsFlat)
	
	return StepResult{
		Params:                paramsFlat,
		Objective:             currentObj,
		GradientNorm:          gradNorm,
		BestObjective:         pb.BestObjective,
		StepsSinceImprovement: pb.StepsSinceImprovement,
		ConsecutiveStuck:      pb.ConsecutiveStuck,
		MinStuckDistance:      minDistance,
		TooCloseToStuck:       tooClose,
		NeedsReset:            needsReset,
		IsJump:                isJump,
		JumpType:              jumpType,
		ClimbState:            pb.ClimbState.String(),
		ClimbSteps:            pb.ClimbSteps,
		InEscapeClimb:         pb.ClimbState == EscapeClimb,
		AdaptiveLR:            pb.Config.LearningRate,
		NoiseScale:            pb.Config.NoiseScale,
	}
}

// updateClimbState updates the hill-climb state machine
func (pb *PathBasedPBit) updateClimbState(currentObj float32) {
	if !pb.Config.EnableHillClimb {
		return
	}
	
	switch pb.ClimbState {
	case Normal:
		if pb.ConsecutiveStuck >= pb.Config.HillClimbTriggerStuck && 
			pb.StepsSinceImprovement > 50 {
			pb.ClimbState = EscapeClimb
			pb.ClimbStartObjective = currentObj
			pb.ClimbWorstObjective = currentObj
			pb.ClimbSteps = 0
			pb.HillClimbCount++
		}
		
	case EscapeClimb:
		pb.ClimbSteps++
		if currentObj > pb.ClimbWorstObjective {
			pb.ClimbWorstObjective = currentObj
		}
		
		timeout := pb.ClimbSteps >= pb.Config.HillClimbMaxSteps
		relativeImprovement := (pb.ClimbStartObjective - currentObj) / 
			(abs(pb.ClimbStartObjective) + 1e-10)
		significantImprovement := relativeImprovement > 0.1
		
		if timeout || significantImprovement {
			pb.ClimbState = Normal
			pb.ClimbStartObjective = float32(math.Inf(1))
			pb.ClimbWorstObjective = float32(math.Inf(1))
			pb.ClimbSteps = 0
		}
	}
}

// computeDelta computes parameter delta using differential pair probability
func (pb *PathBasedPBit) computeDelta(gradient [][]float32, gradNorm, currentObj float32) [][]float32 {
	delta := make([][]float32, pb.Config.NumNodes)
	
	isSmallGrad := gradNorm < 1e-6
	
	for i := range delta {
		delta[i] = make([]float32, pb.Config.NumBits)
		
		for j := range delta[i] {
			// Compute effective field with coupling
			effectiveGrad := gradient[i][j]
			
			// Compute differential pair probability
			vDiff := abs(effectiveGrad) / 25.0
			prob := pb.differentialPairProbability(vDiff)
			
			// Compute step based on climb state
			if pb.ClimbState == EscapeClimb {
				// Hill-climb mode: move opposite to gradient
				var direction float32
				if isSmallGrad {
					direction = float32(pb.RNG.NormFloat64())
				} else {
					direction = gradient[i][j] / (gradNorm + 1e-10)
				}
				
				stepSize := pb.Config.LearningRate * pb.Config.HillClimbRewardScale
				if isSmallGrad {
					stepSize *= 1.5
				}
				delta[i][j] = direction * stepSize
			} else {
				// Normal mode: gradient descent
				adaptiveLR := pb.Config.LearningRate
				if gradNorm < 1e-4 && currentObj < 1.0 {
					adaptiveLR *= 0.1
				} else if pb.StepsSinceImprovement > 100 {
					adaptiveLR *= 2.0
				}
				
				stepSize := prob * adaptiveLR * abs(effectiveGrad)
				delta[i][j] = -stepSize * sign(effectiveGrad)
			}
			
			// Add noise
			noise := float32(pb.RNG.NormFloat64()) * pb.Config.Gamma * 0.03 * abs(delta[i][j])
			delta[i][j] += noise
			
			// Clip
			delta[i][j] = clip(delta[i][j], pb.Config.ClipDelta[0], pb.Config.ClipDelta[1])
		}
	}
	
	return delta
}

// differentialPairProbability computes probability from differential pair model
func (pb *PathBasedPBit) differentialPairProbability(vDiff float32) float32 {
	eta := float32(pb.RNG.NormFloat64()) * float32(math.Sqrt(float64(pb.NoisePower)))
	vEff := pb.Config.Alpha*vDiff + pb.Config.Beta*pb.Config.ITail + pb.Config.Gamma*eta
	arg := clip(-vEff/pb.VTKappa, -500.0, 500.0)
	prob := 1.0 / (1.0 + float32(math.Exp(float64(arg))))
	return prob
}

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

func euclideanDistance(a, b []float32) float32 {
	sum := float32(0.0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func norm(v []float32) float32 {
	sum := float32(0.0)
	for _, val := range v {
		sum += val * val
	}
	return float32(math.Sqrt(float64(sum)))
}

func clip(val, min, max float32) float32 {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}

func abs(val float32) float32 {
	if val < 0 {
		return -val
	}
	return val
}

func sign(val float32) float32 {
	if val < 0 {
		return -1
	}
	if val > 0 {
		return 1
	}
	return 0
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func flattenTo(src [][]float32, dst []float32) {
	idx := 0
	for i := range src {
		for j := range src[i] {
			dst[idx] = src[i][j]
			idx++
		}
	}
}

func unflattenFrom(src []float32, dst [][]float32) {
	idx := 0
	for i := range dst {
		for j := range dst[i] {
			dst[i][j] = src[idx]
			idx++
		}
	}
}

// ================================================================
// EXAMPLE USAGE
// ================================================================

// Example demonstrates basic usage
func Example() {
//func main() {
	// Define problem
	problemDim := 10
	
	// Create configs
	sarConfig := DefaultSARConfig()
	sarConfig.EnableJumps = true
	
	pbitConfig := DefaultPBitConfig()
	pbitConfig.EnableHillClimb = true
	pbitConfig.EnableQuantumJumps = true
	
	// Create optimizer
	memoryManager := NewSARMemoryManager(problemDim, sarConfig)
	optimizer := NewPathBasedPBit(problemDim, pbitConfig, memoryManager)
	
	// Define objective: Sphere function
	objectiveFn := func(params []float32) float32 {
		sum := float32(0.0)
		for _, p := range params {
			sum += p * p
		}
		return sum
	}
	
	// Define gradient
	gradientFn := func(params []float32) []float32 {
		grad := make([]float32, len(params))
		for i := range params {
			grad[i] = 2 * params[i]
		}
		return grad
	}
	
	// Optimize
	for step := 0; step < 1000; step++ {
		result := optimizer.Step(gradientFn, objectiveFn, 100)
		
		if step%100 == 0 {
			println("Step:", step, "Objective:", result.BestObjective)
		}
	}
}
