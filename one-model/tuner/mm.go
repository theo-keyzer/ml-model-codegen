// Package mmsim implements the MM.rio schema simulator
// Parses JSON-formatted rio data files
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ================================================================
// JSON STRUCTURES
// ================================================================

type JSONReference struct {
	Type string `json:"type"`
	ID   int    `json:"id"`
}

type JSONPayload map[string]interface{}

type JSONCanon struct {
	Me      JSONReference   `json:"me"`
	Link    []int           `json:"Link"`
	Section []int           `json:"Section"`
	Payload JSONPayload     `json:"payload"`
}

type JSONLink struct {
	Me       JSONReference `json:"me"`
	Parent   JSONReference `json:"parent"`
	Concept  JSONReference `json:"concept"`
	Relation JSONReference `json:"relation"`
	Payload  JSONPayload   `json:"payload"`
}

type JSONSection struct {
	Me      JSONReference `json:"me"`
	Parent  JSONReference `json:"parent"`
	Payload JSONPayload   `json:"payload"`
}

type JSONObjective struct {
	Me            JSONReference   `json:"me"`
	CanonicalForm JSONReference   `json:"canonical_form"`
	Memory        []int           `json:"Memory"`
	Thought       []int           `json:"Thought"`
	Payload       JSONPayload     `json:"payload"`
}

type JSONMemory struct {
	Me            JSONReference `json:"me"`
	Parent        JSONReference `json:"parent"`
	CanonicalForm JSONReference `json:"canonical_form"`
	Payload       JSONPayload   `json:"payload"`
}

type JSONThought struct {
	Me            JSONReference `json:"me"`
	Parent        JSONReference `json:"parent"`
	CanonicalForm JSONReference `json:"canonical_form"`
	Memo          []int         `json:"Memo"`
	Payload       JSONPayload   `json:"payload"`
}

type JSONMemo struct {
	Me            JSONReference `json:"me"`
	Parent        JSONReference `json:"parent"`
	CanonicalForm JSONReference `json:"canonical_form"`
	ObjectiveID   JSONReference `json:"objective_id"`
	MemoryID      JSONReference `json:"memory_id"`
	Payload       JSONPayload   `json:"payload"`
}

type JSONOps struct {
	Me      JSONReference `json:"me"`
	Payload JSONPayload   `json:"payload"`
}

type JSONDocument struct {
	Canon     []JSONCanon     `json:"Canon"`
	Link      []JSONLink      `json:"Link"`
	Section   []JSONSection   `json:"Section"`
	Objective []JSONObjective `json:"Objective"`
	Memory    []JSONMemory    `json:"Memory"`
	Thought   []JSONThought   `json:"Thought"`
	Memo      []JSONMemo      `json:"Memo"`
	Ops       []JSONOps       `json:"Ops"`
}

// ================================================================
// SCHEMA DEFINITIONS
// ================================================================

type ElementType string

const (
	TypeKey      ElementType = "key"
	TypeWord     ElementType = "word"
	TypeText     ElementType = "text"
	TypeNumber   ElementType = "number"
	TypeTree     ElementType = "tree"
	TypeRef      ElementType = "ref"
	TypeLink     ElementType = "link"
	TypeTypeOf   ElementType = "type_of"
	TypeUpCopy   ElementType = "up_copy"
	TypeRefShare ElementType = "ref_share"
	TypeRefCopy  ElementType = "ref_copy"
	TypeRefChild ElementType = "ref_child"
)

type CheckType string

const (
	CheckDerived  CheckType = "-"
	CheckDot      CheckType = "."
	CheckRequired CheckType = "+"
	CheckOne      CheckType = "1"
	CheckOptional CheckType = "*"
)

type FindType string

const (
	FindTop  FindType = "Find"
	FindIn   FindType = "FindIn"
	FindNone FindType = "."
)

type Element struct {
	Name    string
	Type    ElementType
	RefComp string
	Check   CheckType
	Doc     string
}

type Comp struct {
	Name     string
	Parent   string
	Find     FindType
	Doc      string
	Elements []Element
}

type Schema struct {
	Components map[string]*Comp
	TopLevel   []string
	ChildTypes map[string][]string
}

// ================================================================
// DATA NODE REPRESENTATION
// ================================================================

type FieldValue struct {
	Raw       string
	Number    float64
	IsNumber  bool
	RefTarget string
}

type Node struct {
	CompType   string
	ID         string
	ParentRef  string
	Fields     map[string]FieldValue
	LineNumber int
	Children   []*Node
	Parent     *Node
}

// ================================================================
// GRAPH
// ================================================================

type Graph struct {
	Schema     *Schema
	Nodes      []*Node
	ByType     map[string][]*Node
	ByID       map[string]map[string]*Node
	Errors     []ValidationError
}

type ValidationError struct {
	Node    *Node
	Field   string
	Message string
	Line    int
}

func (e ValidationError) String() string {
	if e.Node != nil {
		return fmt.Sprintf("node '%s' at line %d: %s", e.Node.ID, e.Line, e.Message)
	}
	return fmt.Sprintf("line %d: %s", e.Line, e.Message)
}

// ================================================================
// JSON PARSER
// ================================================================

func ParseJSONData(r io.Reader) (*Graph, error) {
	var doc JSONDocument
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&doc); err != nil {
		return nil, fmt.Errorf("JSON decode error: %w", err)
	}

	graph := &Graph{
		Schema: createMinimalSchema(),
		Nodes:  make([]*Node, 0),
		ByType: make(map[string][]*Node),
		ByID:   make(map[string]map[string]*Node),
		Errors: make([]ValidationError, 0),
	}

	// Parse Canons
	for _, canon := range doc.Canon {
		node := createNodeFromPayload("Canon", canon.Payload)
		graph.addNode(node)
	}

	// Parse Objectives
	for _, obj := range doc.Objective {
		node := createNodeFromPayload("Objective", obj.Payload)
		graph.addNode(node)
	}

	// Parse Thoughts
	for _, thought := range doc.Thought {
		node := createNodeFromPayload("Thought", thought.Payload)
		if thought.Parent.Type == "Objective" {
			node.ParentRef = fmt.Sprintf("obj_%d", thought.Parent.ID)
		}
		graph.addNode(node)
	}

	// Parse Memos
	for _, memo := range doc.Memo {
		node := createNodeFromPayload("Memo", memo.Payload)
		if memo.Parent.Type == "Thought" {
			node.ParentRef = fmt.Sprintf("thought_%d", memo.Parent.ID)
		}
		graph.addNode(node)
	}

	// Parse Memories
	for _, mem := range doc.Memory {
		node := createNodeFromPayload("Memory", mem.Payload)
		if mem.Parent.Type == "Objective" {
			node.ParentRef = fmt.Sprintf("obj_%d", mem.Parent.ID)
		}
		graph.addNode(node)
	}

	// Parse Ops
	for _, ops := range doc.Ops {
		node := createNodeFromPayload("Ops", ops.Payload)
		graph.addNode(node)
	}

	return graph, nil
}

func createNodeFromPayload(compType string, payload JSONPayload) *Node {
	node := &Node{
		CompType: compType,
		Fields:   make(map[string]FieldValue),
	}

	for key, value := range payload {
		if key == "_key" || key == "_lno" || key == "kParentp" {
			continue
		}

		strValue := fmt.Sprintf("%v", value)
		fv := FieldValue{Raw: strValue}

		// Try to parse as number
		switch v := value.(type) {
		case float64:
			fv.Number = v
			fv.IsNumber = true
		case int:
			fv.Number = float64(v)
			fv.IsNumber = true
		case bool:
			if v {
				fv.Raw = "true"
			} else {
				fv.Raw = "false"
			}
		case string:
			fv.Raw = v
		}

		node.Fields[key] = fv

		// Set ID if this is a key field
		if isKeyField(compType, key) {
			node.ID = fv.Raw
		}
	}

	return node
}

func isKeyField(compType, fieldName string) bool {
	keyFields := map[string]string{
		"Canon":     "name",
		"Objective": "objective_id",
		"Thought":   "thought_id",
		"Memo":      "memo_id",
		"Memory":    "memory_id",
		"Ops":       "ops_id",
	}
	return keyFields[compType] == fieldName
}

func (g *Graph) addNode(node *Node) {
	g.Nodes = append(g.Nodes, node)

	if g.ByType[node.CompType] == nil {
		g.ByType[node.CompType] = make([]*Node, 0)
	}
	g.ByType[node.CompType] = append(g.ByType[node.CompType], node)

	if g.ByID[node.CompType] == nil {
		g.ByID[node.CompType] = make(map[string]*Node)
	}
	if node.ID != "" && node.ID != "." {
		g.ByID[node.CompType][node.ID] = node
	}
}

func createMinimalSchema() *Schema {
	return &Schema{
		Components: make(map[string]*Comp),
		TopLevel:   []string{"Canon", "Objective", "Ops"},
		ChildTypes: map[string][]string{
			"Objective": {"Thought", "Memory"},
			"Thought":   {"Memo"},
		},
	}
}

// ================================================================
// VALIDATION
// ================================================================

func (g *Graph) Validate() {
	g.Errors = make([]ValidationError, 0)
	// Basic validation - can be extended
}

// ================================================================
// GRAPH BUILDING
// ================================================================

func (g *Graph) BuildTree() {
	// Link Thoughts to Objectives
	for _, thought := range g.ByType["Thought"] {
		if thought.ParentRef != "" {
			for _, obj := range g.ByType["Objective"] {
				objRef := fmt.Sprintf("obj_%d", 0) // Simplified - would need proper ID mapping
				if thought.ParentRef == objRef || matchesObjectiveID(thought, obj) {
					thought.Parent = obj
					obj.Children = append(obj.Children, thought)
					break
				}
			}
		}
	}

	// Link Memos to Thoughts
	for _, memo := range g.ByType["Memo"] {
		if memo.ParentRef != "" {
			for _, thought := range g.ByType["Thought"] {
				if matchesThoughtID(memo, thought) {
					memo.Parent = thought
					thought.Children = append(thought.Children, memo)
					break
				}
			}
		}
	}

	// Link Memories to Objectives
	for _, mem := range g.ByType["Memory"] {
		if mem.ParentRef != "" {
			for _, obj := range g.ByType["Objective"] {
				if matchesObjectiveID(mem, obj) {
					mem.Parent = obj
					obj.Children = append(obj.Children, mem)
					break
				}
			}
		}
	}
}

func matchesObjectiveID(node, obj *Node) bool {
	objID := obj.Fields["objective_id"].Raw
	return objID != "" && (node.ParentRef == objID || 
		strings.Contains(node.ParentRef, objID))
}

func matchesThoughtID(node, thought *Node) bool {
	thoughtID := thought.Fields["thought_id"].Raw
	return thoughtID != "" && strings.Contains(node.ParentRef, thoughtID)
}

// ================================================================
// QUERY INTERFACE
// ================================================================

func (g *Graph) GetObjectives() []*Node {
	return g.ByType["Objective"]
}

func (g *Graph) GetThoughts(obj *Node) []*Node {
	thoughts := make([]*Node, 0)
	for _, child := range obj.Children {
		if child.CompType == "Thought" {
			thoughts = append(thoughts, child)
		}
	}
	sort.Slice(thoughts, func(i, j int) bool {
		ti := thoughts[i].Fields["tree_linear"]
		tj := thoughts[j].Fields["tree_linear"]
		return ti.Number < tj.Number
	})
	return thoughts
}

func (g *Graph) GetMemos(thought *Node) []*Node {
	memos := make([]*Node, 0)
	for _, child := range thought.Children {
		if child.CompType == "Memo" {
			memos = append(memos, child)
		}
	}
	return memos
}

func (g *Graph) GetMemories(obj *Node) []*Node {
	memories := make([]*Node, 0)
	for _, child := range obj.Children {
		if child.CompType == "Memory" {
			memories = append(memories, child)
		}
	}
	return memories
}

func (g *Graph) GetCanons() []*Node {
	return g.ByType["Canon"]
}

func (g *Graph) GetCanonsByCategory() map[string][]*Node {
	result := make(map[string][]*Node)
	for _, canon := range g.ByType["Canon"] {
		cat := canon.Fields["category"].Raw
		if cat == "" {
			cat = "."
		}
		result[cat] = append(result[cat], canon)
	}
	return result
}

func (g *Graph) GetOps() []*Node {
	return g.ByType["Ops"]
}

// ================================================================
// SIMULATOR ENGINE
// ================================================================

type SimulatorStats struct {
	StepsExecuted  int
	OpsExecuted    int
	MemoriesUsed   int
	CanonsCovered  map[string]bool
}

type Simulator struct {
	Graph       *Graph
	Stats       SimulatorStats
	CurrentObj  *Node
	OpHandlers  map[string]func(*Node) error
	Output      io.Writer
}

func NewSimulator(g *Graph) *Simulator {
	return &Simulator{
		Graph:      g,
		Stats:      SimulatorStats{CanonsCovered: make(map[string]bool)},
		OpHandlers: make(map[string]func(*Node) error),
		Output:     os.Stdout,
	}
}

func (s *Simulator) RegisterOpHandler(kind string, handler func(*Node) error) {
	s.OpHandlers[kind] = handler
}

func (s *Simulator) Run() error {
	objectives := s.Graph.GetObjectives()
	
	for _, obj := range objectives {
		if err := s.runObjective(obj); err != nil {
			return err
		}
	}
	
	return nil
}

func (s *Simulator) runObjective(obj *Node) error {
	s.CurrentObj = obj
	fmt.Fprintf(s.Output, "Starting objective: %s\n", obj.ID)
	
	thoughts := s.Graph.GetThoughts(obj)
	
	for i, thought := range thoughts {
		s.Stats.StepsExecuted++
		
		thoughtType := thought.Fields["type"].Raw
		fmt.Fprintf(s.Output, "  Step %d: Executing thought '%s' (type: %s)\n",
			i+1, thought.ID, thoughtType)
		
		if cf := thought.Fields["canonical_form"]; cf.Raw != "" && cf.Raw != "." {
			s.Stats.CanonsCovered[cf.Raw] = true
		}
		
		memos := s.Graph.GetMemos(thought)
		for _, memo := range memos {
			fmt.Fprintf(s.Output, "    Memo: %s\n", memo.ID)
		}
	}
	
	memories := s.Graph.GetMemories(obj)
	for _, mem := range memories {
		s.Stats.MemoriesUsed++
		fmt.Fprintf(s.Output, "  Processing memory: %s\n", mem.ID)
	}
	
	fmt.Fprintf(s.Output, "Completed objective: %s\n", obj.ID)
	return nil
}

// ================================================================
// GRAPH VISUALIZATION
// ================================================================

func (g *Graph) PrintSummary(w io.Writer) {
	fmt.Fprintf(w, "=== MM.rio Graph Summary ===\n\n")
	
	fmt.Fprintf(w, "Component Counts:\n")
	types := make([]string, 0, len(g.ByType))
	for t := range g.ByType {
		types = append(types, t)
	}
	sort.Slice(types, func(i, j int) bool {
		return len(g.ByType[types[i]]) > len(g.ByType[types[j]])
	})
	for _, t := range types {
		fmt.Fprintf(w, "  %s: %d\n", t, len(g.ByType[t]))
	}
	
	objectives := g.GetObjectives()
	fmt.Fprintf(w, "\nObjectives (%d):\n", len(objectives))
	for _, obj := range objectives {
		thoughts := g.GetThoughts(obj)
		memories := g.GetMemories(obj)
		fmt.Fprintf(w, "  - %s: %s\n", obj.ID, obj.Fields["doc"].Raw)
		fmt.Fprintf(w, "    Thoughts: %d, Memories: %d\n", len(thoughts), len(memories))
	}
	
	canonsByCat := g.GetCanonsByCategory()
	total := len(g.ByType["Canon"])
	fmt.Fprintf(w, "\nCanons (%d) by Category:\n", total)
	for cat, canons := range canonsByCat {
		fmt.Fprintf(w, "  %s: %d\n", cat, len(canons))
	}
	
	ops := g.GetOps()
	fmt.Fprintf(w, "\nOperations (%d):\n", len(ops))
	for _, op := range ops {
		fmt.Fprintf(w, "  - %s [%s]: %s\n", op.ID, op.Fields["kind"].Raw, op.Fields["doc"].Raw)
	}
}

func (g *Graph) PrintObjectiveTree(w io.Writer) {
	fmt.Fprintf(w, "\nObjective tree walk:\n")
	
	for _, obj := range g.GetObjectives() {
		fmt.Fprintf(w, "Objective: %s\n", obj.ID)
		
		thoughts := g.GetThoughts(obj)
		for _, thought := range thoughts {
			fmt.Fprintf(w, "  Thought: %s\n", thought.ID)
			
			memos := g.GetMemos(thought)
			for _, memo := range memos {
				fmt.Fprintf(w, "    Memo: %s\n", memo.ID)
			}
		}
		
		memories := g.GetMemories(obj)
		for _, mem := range memories {
			fmt.Fprintf(w, "  Memory: %s\n", mem.ID)
		}
	}
}

// ================================================================
// PBIT INTEGRATION
// ================================================================

type PBitRunner struct {
	Graph     *Graph
	Optimizer *PathBasedPBit
	Config    PBitConfig
	Results   map[string][]float32
}

func NewPBitRunner(g *Graph, problemDim int) *PBitRunner {
	config := DefaultPBitConfig()
	
	for _, canon := range g.ByType["Canon"] {
		if canon.Fields["category"].Raw == "config_value" {
			parseConfigFromCanon(canon, &config)
		}
	}
	
	return &PBitRunner{
		Graph:     g,
		Config:    config,
		Optimizer: NewPathBasedPBit(problemDim, config, nil),
		Results:   make(map[string][]float32),
	}
}

func parseConfigFromCanon(canon *Node, config *PBitConfig) {
	if lr := canon.Fields["learning_rate"]; lr.IsNumber {
		config.LearningRate = float32(lr.Number)
	}
	if mb := canon.Fields["momentum_beta"]; mb.IsNumber {
		config.MomentumBeta = float32(mb.Number)
	}
	if ns := canon.Fields["noise_scale"]; ns.IsNumber {
		config.NoiseScale = float32(ns.Number)
	}
	if ej := canon.Fields["enable_quantum_jumps"]; ej.Raw == "true" {
		config.EnableQuantumJumps = true
	}
	if ec := canon.Fields["enable_hill_climb"]; ec.Raw == "true" {
		config.EnableHillClimb = true
	}
}

func (r *PBitRunner) RunOptimization(problemFn string, steps int, output io.Writer) {
	objectiveFn, gradientFn := r.getObjectiveFunction(problemFn)
	
	fmt.Fprintf(output, "Running %s optimization for %d steps...\n", problemFn, steps)
	
	for step := 0; step < steps; step++ {
		result := r.Optimizer.Step(gradientFn, objectiveFn, 100)
		
		if step%100 == 0 {
			fmt.Fprintf(output, "  Step %d: objective=%.6f, grad_norm=%.4f\n",
				step, result.Objective, result.GradientNorm)
		}
	}
	
	fmt.Fprintf(output, "\nFinal objective: %.8f\n", r.Optimizer.BestObjective)
}

func (r *PBitRunner) getObjectiveFunction(name string) (func([]float32) float32, func([]float32) []float32) {
	switch name {
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

// ================================================================
// BENCHMARK FUNCTIONS
// ================================================================

func sphereObjective(params []float32) float32 {
	sum := float32(0.0)
	for _, p := range params {
		sum += p * p
	}
	return sum
}

func sphereGradient(params []float32) []float32 {
	grad := make([]float32, len(params))
	for i := range params {
		grad[i] = 2 * params[i]
	}
	return grad
}

func rastriginObjective(params []float32) float32 {
	n := float32(len(params))
	sum := 10 * n
	for _, p := range params {
		sum += p*p - 10*float32(math.Cos(float64(2*math.Pi*p)))
	}
	return sum
}

func rastriginGradient(params []float32) []float32 {
	grad := make([]float32, len(params))
	for i, p := range params {
		grad[i] = 2*p + 20*float32(math.Pi)*float32(math.Sin(float64(2*math.Pi*p)))
	}
	return grad
}

func ackleyObjective(params []float32) float32 {
	n := float64(len(params))
	sumSq := float64(0)
	sumCos := float64(0)
	
	for _, p := range params {
		sumSq += float64(p * p)
		sumCos += math.Cos(2 * math.Pi * float64(p))
	}
	
	term1 := -20 * math.Exp(-0.2*math.Sqrt(sumSq/n))
	term2 := -math.Exp(sumCos / n)
	
	return float32(term1 + term2 + 20 + math.E)
}

func ackleyGradient(params []float32) []float32 {
	grad := make([]float32, len(params))
	n := float64(len(params))
	
	sumSq := float64(0)
	for _, p := range params {
		sumSq += float64(p * p)
	}
	
	sqrtTerm := math.Sqrt(sumSq / n)
	expTerm := math.Exp(-0.2 * sqrtTerm)
	
	for i, p := range params {
		if sqrtTerm > 1e-10 {
			grad[i] = float32(4.0 * float64(p) * expTerm / (sqrtTerm * n))
		}
		grad[i] += float32(2 * math.Pi * math.Sin(2*math.Pi*float64(p)) / n)
	}
	return grad
}

func extractFloat(s, key string) float64 {
	pattern := regexp.MustCompile(key + `"?\s*:\s*([0-9.]+)`)
	matches := pattern.FindStringSubmatch(s)
	if len(matches) > 1 {
		if val, err := strconv.ParseFloat(matches[1], 64); err == nil {
			return val
		}
	}
	return 0
}

// ================================================================
// MAIN
// ================================================================

func main() {
	// Open JSON file
	if len(os.Args) > 1 && os.Args[1] == "explore" {
		// Run full configuration exploration
		stepsPerRun := 1000
		if len(os.Args) > 2 {
			fmt.Sscanf(os.Args[2], "%d", &stepsPerRun)
		}
		RunConfigurationExploration("mm.json", stepsPerRun)
		return
	}
	file, err := os.Open("mm.json")
	if err != nil {
		fmt.Printf("Error opening mm.json: %v\n", err)
		return
	}
	defer file.Close()

	fmt.Printf("=== MM.rio Simulator (JSON) ===\n\n")

	// Parse JSON data
	graph, err := ParseJSONData(file)
	if err != nil {
		fmt.Printf("Data parse error: %v\n", err)
		return
	}
	fmt.Printf("Parsed %d nodes\n\n", len(graph.Nodes))

	// Build tree and validate
	graph.BuildTree()
	graph.Validate()

	// Print summary
	graph.PrintSummary(os.Stdout)

	// Run simulation
	fmt.Printf("\n=== Running Simulation ===\n\n")
	sim := NewSimulator(graph)
	
	sim.RegisterOpHandler("think", func(op *Node) error {
		fmt.Printf("      [THINK] %s\n", op.ID)
		return nil
	})
	sim.RegisterOpHandler("manifest", func(op *Node) error {
		fmt.Printf("      [MANIFEST] %s\n", op.ID)
		return nil
	})
	
	if err := sim.Run(); err != nil {
		fmt.Printf("Simulation error: %v\n", err)
	}
	
	fmt.Printf("\nSimulation completed. Steps: %d, Ops executed: %d\n",
		sim.Stats.StepsExecuted, sim.Stats.OpsExecuted)

	// Print query examples
	fmt.Printf("\n=== Query Examples ===\n")
	
	canonsByCat := graph.GetCanonsByCategory()
	fmt.Printf("\nCanons by category:\n")
	for cat, canons := range canonsByCat {
		fmt.Printf("  %s: %d\n", cat, len(canons))
	}
	
	// Print objective tree
	graph.PrintObjectiveTree(os.Stdout)

	// Run PBit optimization demo
	fmt.Printf("\n=== PBit Optimization Demo ===\n\n")
	
	runner := NewPBitRunner(graph, 10)
	runner.RunOptimization("sphere", 500, os.Stdout)
}
