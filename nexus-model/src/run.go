package main

import (
	"strings"
	"fmt"
	"strconv"
)

type ActT struct {
	index       map[string]int
	ApProject [] *KpProject
	ApComputeGraph [] *KpComputeGraph
	ApHardwareTarget [] *KpHardwareTarget
	ApDataTensor [] *KpDataTensor
	ApTensorConsumer [] *KpTensorConsumer
	ApOperation [] *KpOperation
	ApOperationArg [] *KpOperationArg
	ApOpDependency [] *KpOpDependency
	ApClassicalOp [] *KpClassicalOp
	ApSpikingOp [] *KpSpikingOp
	ApPlasticityRule [] *KpPlasticityRule
	ApAnalogOp [] *KpAnalogOp
	ApQuantumOp [] *KpQuantumOp
	ApQubitTarget [] *KpQubitTarget
	ApControlQubit [] *KpControlQubit
	ApQuantumCircuit [] *KpQuantumCircuit
	ApPhotonicOp [] *KpPhotonicOp
	ApMolecularOp [] *KpMolecularOp
	ApReactant [] *KpReactant
	ApProduct [] *KpProduct
	ApHybridOp [] *KpHybridOp
	ApFallbackMode [] *KpFallbackMode
	ApRedundancyStrategy [] *KpRedundancyStrategy
	ApProfilingHook [] *KpProfilingHook
	ApAdaptiveParameter [] *KpAdaptiveParameter
	ApSecureCompute [] *KpSecureCompute
	ApSearchSpace [] *KpSearchSpace
	ApSearchTarget [] *KpSearchTarget
	ApSearchParameter [] *KpSearchParameter
	ApEnergyBudget [] *KpEnergyBudget
	ApEnergyAllocation [] *KpEnergyAllocation
	ApTileMapping [] *KpTileMapping
	ApTileTarget [] *KpTileTarget
	ApPrivacyBudget [] *KpPrivacyBudget
	ApTraceCollection [] *KpTraceCollection
	ApEvolvableGraph [] *KpEvolvableGraph
	ApProvenance [] *KpProvenance
	ApOptimizationRun [] *KpOptimizationRun
	ApValidation [] *KpValidation
	ApGenotype [] *KpGenotype
	ApPhenotype [] *KpPhenotype
	ApHardware [] *KpHardware
	ApPhysicsModel [] *KpPhysicsModel
	ApClassicalHardware [] *KpClassicalHardware
	ApNeuromorphicHardware [] *KpNeuromorphicHardware
	ApAnalogHardware [] *KpAnalogHardware
	ApQuantumHardware [] *KpQuantumHardware
	ApPhotonicHardware [] *KpPhotonicHardware
	ApMolecularHardware [] *KpMolecularHardware
	ApPowerDomain [] *KpPowerDomain
	ApSpatialArray [] *KpSpatialArray
	ApKernel [] *KpKernel
	ApFusionPattern [] *KpFusionPattern
	ApFusionHardwareTarget [] *KpFusionHardwareTarget
	ApFusionOpTarget [] *KpFusionOpTarget
	ApDataFormatConverter [] *KpDataFormatConverter
	ApOptimizationStrategy [] *KpOptimizationStrategy
	ApFitnessFunction [] *KpFitnessFunction
	ApFitnessComponent [] *KpFitnessComponent
	ApFaultModel [] *KpFaultModel
	ApSimulator [] *KpSimulator
	ApSimulationLevel [] *KpSimulationLevel
	ApAdaptiveFidelity [] *KpAdaptiveFidelity
	ApDeterminismConstraint [] *KpDeterminismConstraint
	ApConstraint [] *KpConstraint
	ApMetric [] *KpMetric
	ApCheckpoint [] *KpCheckpoint
	ApParadigmRule [] *KpParadigmRule
	ApNeuronModelRule [] *KpNeuronModelRule
	ApQuantumGateRule [] *KpQuantumGateRule
	ApDataflowRule [] *KpDataflowRule
	ApFusionStrategyRule [] *KpFusionStrategyRule
	ApExtension [] *KpExtension
	ApCustomParadigm [] *KpCustomParadigm
	ApActor [] *KpActor
	ApAll [] *KpAll
	ApDu [] *KpDu
	ApNew [] *KpNew
	ApRefs [] *KpRefs
	ApVar [] *KpVar
	ApIts [] *KpIts
	ApC [] *KpC
	ApCs [] *KpCs
	ApOut [] *KpOut
	ApIn [] *KpIn
	ApBreak [] *KpBreak
	ApAdd [] *KpAdd
	ApThis [] *KpThis
	ApReplace [] *KpReplace
}

func refs(act *ActT) int {
	errs := 0
	v := ""
	p := -1
	res := 0
	err := false
	for _, st := range act.ApComputeGraph {

//  nexus.unit:29, g_runh.act:180

		v, _ = st.Names["project_ref"]
		err, res = fnd3(act, "Project_" + v, v, "ref:ComputeGraph.project_ref:Project." + v,  "+", st.LineNo, "nexus.unit:29, g_runh.act:184" );
		st.Kproject_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApHardwareTarget {

//  nexus.unit:35, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:HardwareTarget.hardware:Hardware." + v,  "+", st.LineNo, "nexus.unit:35, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOperation {

//  nexus.unit:69, g_runh.act:180

		v, _ = st.Names["hardware_target"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Operation.hardware_target:Hardware." + v,  "*", st.LineNo, "nexus.unit:69, g_runh.act:184" );
		st.Khardware_targetp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:70, g_runh.act:180

		v, _ = st.Names["kernel_ref"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:Operation.kernel_ref:Kernel." + v,  "*", st.LineNo, "nexus.unit:70, g_runh.act:184" );
		st.Kkernel_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApClassicalOp {

//  nexus.unit:103, g_runh.act:180

		v, _ = st.Names["fusion_pattern"]
		err, res = fnd3(act, "FusionPattern_" + v, v, "ref:ClassicalOp.fusion_pattern:FusionPattern." + v,  "*", st.LineNo, "nexus.unit:103, g_runh.act:184" );
		st.Kfusion_patternp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApAnalogOp {

//  nexus.unit:144, g_runh.act:180

		v, _ = st.Names["array_target"]
		err, res = fnd3(act, "AnalogHardware_" + v, v, "ref:AnalogOp.array_target:AnalogHardware." + v,  "+", st.LineNo, "nexus.unit:144, g_runh.act:184" );
		st.Karray_targetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApRedundancyStrategy {

//  nexus.unit:255, g_runh.act:180

		v, _ = st.Names["checkpoints"]
		err, res = fnd3(act, "Checkpoint_" + v, v, "ref:RedundancyStrategy.checkpoints:Checkpoint." + v,  "*", st.LineNo, "nexus.unit:255, g_runh.act:184" );
		st.Kcheckpointsp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApProfilingHook {

//  nexus.unit:266, g_runh.act:180

		v, _ = st.Names["metrics_ref"]
		err, res = fnd3(act, "Metric_" + v, v, "ref:ProfilingHook.metrics_ref:Metric." + v,  "*", st.LineNo, "nexus.unit:266, g_runh.act:184" );
		st.Kmetrics_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApEvolvableGraph {

//  nexus.unit:395, g_runh.act:180

		v, _ = st.Names["fitness_metric"]
		err, res = fnd3(act, "Metric_" + v, v, "ref:EvolvableGraph.fitness_metric:Metric." + v,  "+", st.LineNo, "nexus.unit:395, g_runh.act:184" );
		st.Kfitness_metricp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOptimizationRun {

//  nexus.unit:420, g_runh.act:180

		v, _ = st.Names["project_ref"]
		err, res = fnd3(act, "Project_" + v, v, "ref:OptimizationRun.project_ref:Project." + v,  "+", st.LineNo, "nexus.unit:420, g_runh.act:184" );
		st.Kproject_refp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:421, g_runh.act:180

		v, _ = st.Names["strategy_ref"]
		err, res = fnd3(act, "OptimizationStrategy_" + v, v, "ref:OptimizationRun.strategy_ref:OptimizationStrategy." + v,  "+", st.LineNo, "nexus.unit:421, g_runh.act:184" );
		st.Kstrategy_refp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:422, g_runh.act:180

		v, _ = st.Names["best_checkpoint"]
		err, res = fnd3(act, "Checkpoint_" + v, v, "ref:OptimizationRun.best_checkpoint:Checkpoint." + v,  "*", st.LineNo, "nexus.unit:422, g_runh.act:184" );
		st.Kbest_checkpointp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApPhenotype {

//  nexus.unit:455, g_runh.act:209

		v, _ = st.Names["realized_graph"]
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_ComputeGraph_" + v,v, "ref_link:Phenotype.realized_graph:Project." + st.Parent + ".ComputeGraph." + v,  "+", st.LineNo, "nexus.unit:455, g_runh.act:212" );
		st.Krealized_graphp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:456, g_runh.act:209

		v, _ = st.Names["genome_source"]
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Genotype_" + v,v, "ref_link:Phenotype.genome_source:Project." + st.Parent + ".Genotype." + v,  "+", st.LineNo, "nexus.unit:456, g_runh.act:212" );
		st.Kgenome_sourcep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApKernel {

//  nexus.unit:597, g_runh.act:180

		v, _ = st.Names["hardware_target"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Kernel.hardware_target:Hardware." + v,  "+", st.LineNo, "nexus.unit:597, g_runh.act:184" );
		st.Khardware_targetp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:598, g_runh.act:180

		v, _ = st.Names["fusion_source"]
		err, res = fnd3(act, "FusionPattern_" + v, v, "ref:Kernel.fusion_source:FusionPattern." + v,  "*", st.LineNo, "nexus.unit:598, g_runh.act:184" );
		st.Kfusion_sourcep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApFusionPattern {

//  nexus.unit:609, g_runh.act:180

		v, _ = st.Names["fused_kernel_ref"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:FusionPattern.fused_kernel_ref:Kernel." + v,  "+", st.LineNo, "nexus.unit:609, g_runh.act:184" );
		st.Kfused_kernel_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApFusionHardwareTarget {

//  nexus.unit:615, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:FusionHardwareTarget.hardware:Hardware." + v,  "+", st.LineNo, "nexus.unit:615, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOptimizationStrategy {

//  nexus.unit:659, g_runh.act:180

		v, _ = st.Names["project_ref"]
		err, res = fnd3(act, "Project_" + v, v, "ref:OptimizationStrategy.project_ref:Project." + v,  "*", st.LineNo, "nexus.unit:659, g_runh.act:184" );
		st.Kproject_refp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:662, g_runh.act:180

		v, _ = st.Names["strategy"]
		err, res = fnd3(act, "OptimizationStrategy_" + v, v, "ref:OptimizationStrategy.strategy:OptimizationStrategy." + v,  "+", st.LineNo, "nexus.unit:662, g_runh.act:184" );
		st.Kstrategyp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApFitnessComponent {

//  nexus.unit:678, g_runh.act:180

		v, _ = st.Names["metric"]
		err, res = fnd3(act, "Metric_" + v, v, "ref:FitnessComponent.metric:Metric." + v,  "+", st.LineNo, "nexus.unit:678, g_runh.act:184" );
		st.Kmetricp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSimulator {

//  nexus.unit:717, g_runh.act:180

		v, _ = st.Names["project_ref"]
		err, res = fnd3(act, "Project_" + v, v, "ref:Simulator.project_ref:Project." + v,  "*", st.LineNo, "nexus.unit:717, g_runh.act:184" );
		st.Kproject_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSimulationLevel {

//  nexus.unit:731, g_runh.act:180

		v, _ = st.Names["target_component"]
		err, res = fnd3(act, "Operation_" + v, v, "ref:SimulationLevel.target_component:Operation." + v,  "+", st.LineNo, "nexus.unit:731, g_runh.act:184" );
		st.Ktarget_componentp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApCheckpoint {

//  nexus.unit:793, g_runh.act:180

		v, _ = st.Names["project_ref"]
		err, res = fnd3(act, "Project_" + v, v, "ref:Checkpoint.project_ref:Project." + v,  "+", st.LineNo, "nexus.unit:793, g_runh.act:184" );
		st.Kproject_refp = res
		if (err == false) {
			errs += 1
		}
//  nexus.unit:798, g_runh.act:180

		v, _ = st.Names["project_source"]
		err, res = fnd3(act, "Project_" + v, v, "ref:Checkpoint.project_source:Project." + v,  "+", st.LineNo, "nexus.unit:798, g_runh.act:184" );
		st.Kproject_sourcep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApAll {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:34, g_runh.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApDu {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:46, g_runh.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApIts {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:87, g_runh.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApThis {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:186, g_runh.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTensorConsumer {

//  nexus.unit:55, g_runh.act:262
	p = st.Me
	p = act.ApTensorConsumer[p].Kparentp
	p = act.ApDataTensor[p].Kparentp
	if p >= 0 {
		st.Kgraphp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:TensorConsumer.graph unresolved from text:TensorConsumer.desc:..x %s (+) > nexus.unit:55, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  nexus.unit:56, g_runh.act:224

 
	if st.Kgraphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:TensorConsumer.consumer_op unresolved from up_copy:TensorConsumer.graph:ComputeGraph %s > nexus.unit:56, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Kgraphp].MyName
		v, _ = st.Names["consumer_op"]
		err, res = fnd3(act, strconv.Itoa(st.Kgraphp) + "_Operation_" + v, v, "ref_child:TensorConsumer.consumer_op:ComputeGraph." + parent + "." + v + " from up_copy:TensorConsumer.graph", "+", st.LineNo, "nexus.unit:56, g_runh.act:236")
		st.Kconsumer_opp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApOperationArg {

//  nexus.unit:78, g_runh.act:262
	p = st.Me
	p = act.ApOperationArg[p].Kparentp
	p = act.ApOperation[p].Kparentp
	if p >= 0 {
		st.Kgraphp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:OperationArg.graph unresolved from word:OperationArg.role:..x %s (+) > nexus.unit:78, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  nexus.unit:79, g_runh.act:224

 
	if st.Kgraphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:OperationArg.tensor_ref unresolved from up_copy:OperationArg.graph:ComputeGraph %s > nexus.unit:79, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Kgraphp].MyName
		v, _ = st.Names["tensor_ref"]
		err, res = fnd3(act, strconv.Itoa(st.Kgraphp) + "_DataTensor_" + v, v, "ref_child:OperationArg.tensor_ref:ComputeGraph." + parent + "." + v + " from up_copy:OperationArg.graph", "+", st.LineNo, "nexus.unit:79, g_runh.act:236")
		st.Ktensor_refp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApOpDependency {

//  nexus.unit:91, g_runh.act:262
	p = st.Me
	p = act.ApOpDependency[p].Kparentp
	p = act.ApOperation[p].Kparentp
	if p >= 0 {
		st.Kgraphp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:OpDependency.graph unresolved from text:OpDependency.desc:..x %s (+) > nexus.unit:91, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  nexus.unit:92, g_runh.act:224

 
	if st.Kgraphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:OpDependency.pred_op unresolved from up_copy:OpDependency.graph:ComputeGraph %s > nexus.unit:92, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Kgraphp].MyName
		v, _ = st.Names["pred_op"]
		err, res = fnd3(act, strconv.Itoa(st.Kgraphp) + "_Operation_" + v, v, "ref_child:OpDependency.pred_op:ComputeGraph." + parent + "." + v + " from up_copy:OpDependency.graph", "+", st.LineNo, "nexus.unit:92, g_runh.act:236")
		st.Kpred_opp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSearchTarget {

//  nexus.unit:302, g_runh.act:262
	p = st.Me
	p = act.ApSearchTarget[p].Kparentp
	p = act.ApSearchSpace[p].Kparentp
	if p >= 0 {
		st.Kgraphp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:SearchTarget.graph unresolved from text:SearchTarget.desc:..x %s (+) > nexus.unit:302, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  nexus.unit:303, g_runh.act:224

 
	if st.Kgraphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:SearchTarget.operation unresolved from up_copy:SearchTarget.graph:ComputeGraph %s > nexus.unit:303, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Kgraphp].MyName
		v, _ = st.Names["operation"]
		err, res = fnd3(act, strconv.Itoa(st.Kgraphp) + "_Operation_" + v, v, "ref_child:SearchTarget.operation:ComputeGraph." + parent + "." + v + " from up_copy:SearchTarget.graph", "+", st.LineNo, "nexus.unit:303, g_runh.act:236")
		st.Koperationp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApEnergyAllocation {

//  nexus.unit:337, g_runh.act:262
	p = st.Me
	p = act.ApEnergyAllocation[p].Kparentp
	p = act.ApEnergyBudget[p].Kparentp
	if p >= 0 {
		st.Kgraphp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:EnergyAllocation.graph unresolved from text:EnergyAllocation.desc:..x %s (+) > nexus.unit:337, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  nexus.unit:338, g_runh.act:224

 
	if st.Kgraphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:EnergyAllocation.operation unresolved from up_copy:EnergyAllocation.graph:ComputeGraph %s > nexus.unit:338, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Kgraphp].MyName
		v, _ = st.Names["operation"]
		err, res = fnd3(act, strconv.Itoa(st.Kgraphp) + "_Operation_" + v, v, "ref_child:EnergyAllocation.operation:ComputeGraph." + parent + "." + v + " from up_copy:EnergyAllocation.graph", "+", st.LineNo, "nexus.unit:338, g_runh.act:236")
		st.Koperationp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApTileTarget {

//  nexus.unit:359, g_runh.act:262
	p = st.Me
	p = act.ApTileTarget[p].Kparentp
	p = act.ApTileMapping[p].Kparentp
	if p >= 0 {
		st.Kgraphp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:TileTarget.graph unresolved from text:TileTarget.desc:..x %s (+) > nexus.unit:359, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  nexus.unit:360, g_runh.act:224

 
	if st.Kgraphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:TileTarget.operation unresolved from up_copy:TileTarget.graph:ComputeGraph %s > nexus.unit:360, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Kgraphp].MyName
		v, _ = st.Names["operation"]
		err, res = fnd3(act, strconv.Itoa(st.Kgraphp) + "_Operation_" + v, v, "ref_child:TileTarget.operation:ComputeGraph." + parent + "." + v + " from up_copy:TileTarget.graph", "+", st.LineNo, "nexus.unit:360, g_runh.act:236")
		st.Koperationp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApOptimizationStrategy {

//  nexus.unit:660, g_runh.act:224

 
	if st.Kproject_refp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:OptimizationStrategy.target_graph unresolved from ref:OptimizationStrategy.project_ref:Project %s > nexus.unit:660, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApProject[st.Kproject_refp].MyName
		v, _ = st.Names["target_graph"]
		err, res = fnd3(act, strconv.Itoa(st.Kproject_refp) + "_ComputeGraph_" + v, v, "ref_child:OptimizationStrategy.target_graph:Project." + parent + "." + v + " from ref:OptimizationStrategy.project_ref", "*", st.LineNo, "nexus.unit:660, g_runh.act:236")
		st.Ktarget_graphp = res
		if !err {
			errs += 1
		}
	}
//  nexus.unit:661, g_runh.act:224

 
	if st.Ktarget_graphp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:OptimizationStrategy.search_space unresolved from ref_child:OptimizationStrategy.target_graph:ComputeGraph %s > nexus.unit:661, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Ktarget_graphp].MyName
		v, _ = st.Names["search_space"]
		err, res = fnd3(act, strconv.Itoa(st.Ktarget_graphp) + "_SearchSpace_" + v, v, "ref_child:OptimizationStrategy.search_space:ComputeGraph." + parent + "." + v + " from ref_child:OptimizationStrategy.target_graph", "+", st.LineNo, "nexus.unit:661, g_runh.act:236")
		st.Ksearch_spacep = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSimulator {

//  nexus.unit:718, g_runh.act:224

 
	if st.Kproject_refp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Simulator.target_graph unresolved from ref:Simulator.project_ref:Project %s > nexus.unit:718, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApProject[st.Kproject_refp].MyName
		v, _ = st.Names["target_graph"]
		err, res = fnd3(act, strconv.Itoa(st.Kproject_refp) + "_ComputeGraph_" + v, v, "ref_child:Simulator.target_graph:Project." + parent + "." + v + " from ref:Simulator.project_ref", "*", st.LineNo, "nexus.unit:718, g_runh.act:236")
		st.Ktarget_graphp = res
		if !err {
			errs += 1
		}
	}
//  nexus.unit:719, g_runh.act:224

 
	if st.Ktarget_graphp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Simulator.fault_models unresolved from ref_child:Simulator.target_graph:ComputeGraph %s > nexus.unit:719, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApComputeGraph[st.Ktarget_graphp].MyName
		v, _ = st.Names["fault_models"]
		err, res = fnd3(act, strconv.Itoa(st.Ktarget_graphp) + "_FaultModel_" + v, v, "ref_child:Simulator.fault_models:ComputeGraph." + parent + "." + v + " from ref_child:Simulator.target_graph", "*", st.LineNo, "nexus.unit:719, g_runh.act:236")
		st.Kfault_modelsp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApCheckpoint {

//  nexus.unit:799, g_runh.act:224

 
	if st.Kproject_sourcep < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Checkpoint.graph_snapshot unresolved from ref:Checkpoint.project_source:Project %s > nexus.unit:799, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApProject[st.Kproject_sourcep].MyName
		v, _ = st.Names["graph_snapshot"]
		err, res = fnd3(act, strconv.Itoa(st.Kproject_sourcep) + "_ComputeGraph_" + v, v, "ref_child:Checkpoint.graph_snapshot:Project." + parent + "." + v + " from ref:Checkpoint.project_source", "*", st.LineNo, "nexus.unit:799, g_runh.act:236")
		st.Kgraph_snapshotp = res
		if !err {
			errs += 1
		}
	}
	}
	return(errs)
}

func DoAll(glob *GlobT, va []string, lno string) int {
	if va[0] == "Project" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Project_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProject[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProject[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProject {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ComputeGraph" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ComputeGraph_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApComputeGraph[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApComputeGraph[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApComputeGraph {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "HardwareTarget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HardwareTarget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHardwareTarget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHardwareTarget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHardwareTarget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "DataTensor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DataTensor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDataTensor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDataTensor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDataTensor {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "TensorConsumer" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TensorConsumer_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTensorConsumer[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTensorConsumer[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTensorConsumer {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Operation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Operation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOperation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOperation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOperation {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "OperationArg" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OperationArg_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOperationArg[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOperationArg[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOperationArg {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "OpDependency" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OpDependency_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOpDependency[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOpDependency[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOpDependency {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ClassicalOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ClassicalOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApClassicalOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApClassicalOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApClassicalOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SpikingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SpikingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSpikingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSpikingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSpikingOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "PlasticityRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PlasticityRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPlasticityRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPlasticityRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPlasticityRule {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "AnalogOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AnalogOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAnalogOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAnalogOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAnalogOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "QuantumOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["QuantumOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApQuantumOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApQuantumOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApQuantumOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "QubitTarget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["QubitTarget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApQubitTarget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApQubitTarget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApQubitTarget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ControlQubit" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ControlQubit_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApControlQubit[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApControlQubit[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApControlQubit {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "QuantumCircuit" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["QuantumCircuit_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApQuantumCircuit[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApQuantumCircuit[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApQuantumCircuit {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "PhotonicOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PhotonicOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPhotonicOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPhotonicOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPhotonicOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "MolecularOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MolecularOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMolecularOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMolecularOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMolecularOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Reactant" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Reactant_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApReactant[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApReactant[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApReactant {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Product" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Product_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProduct[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProduct[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProduct {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "HybridOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HybridOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHybridOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHybridOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHybridOp {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FallbackMode" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FallbackMode_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFallbackMode[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFallbackMode[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFallbackMode {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "RedundancyStrategy" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["RedundancyStrategy_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApRedundancyStrategy[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApRedundancyStrategy[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApRedundancyStrategy {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ProfilingHook" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ProfilingHook_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProfilingHook[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProfilingHook[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProfilingHook {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "AdaptiveParameter" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AdaptiveParameter_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAdaptiveParameter[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAdaptiveParameter[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAdaptiveParameter {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SecureCompute" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SecureCompute_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSecureCompute[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSecureCompute[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSecureCompute {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SearchSpace" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchSpace_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchSpace[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchSpace[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchSpace {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SearchTarget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchTarget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchTarget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchTarget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchTarget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SearchParameter" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchParameter_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchParameter[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchParameter[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchParameter {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "EnergyBudget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnergyBudget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnergyBudget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnergyBudget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnergyBudget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "EnergyAllocation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnergyAllocation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnergyAllocation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnergyAllocation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnergyAllocation {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "TileMapping" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TileMapping_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTileMapping[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTileMapping[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTileMapping {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "TileTarget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TileTarget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTileTarget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTileTarget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTileTarget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "PrivacyBudget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PrivacyBudget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPrivacyBudget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPrivacyBudget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPrivacyBudget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "TraceCollection" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TraceCollection_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTraceCollection[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTraceCollection[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTraceCollection {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "EvolvableGraph" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EvolvableGraph_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEvolvableGraph[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEvolvableGraph[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEvolvableGraph {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Provenance" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Provenance_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProvenance[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProvenance[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProvenance {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "OptimizationRun" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OptimizationRun_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOptimizationRun[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOptimizationRun[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOptimizationRun {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Validation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Validation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApValidation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApValidation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApValidation {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Genotype" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Genotype_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApGenotype[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApGenotype[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApGenotype {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Phenotype" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Phenotype_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPhenotype[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPhenotype[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPhenotype {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Hardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Hardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "PhysicsModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PhysicsModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPhysicsModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPhysicsModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPhysicsModel {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ClassicalHardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ClassicalHardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApClassicalHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApClassicalHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApClassicalHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "NeuromorphicHardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuromorphicHardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuromorphicHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuromorphicHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuromorphicHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "AnalogHardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AnalogHardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAnalogHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAnalogHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAnalogHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "QuantumHardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["QuantumHardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApQuantumHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApQuantumHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApQuantumHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "PhotonicHardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PhotonicHardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPhotonicHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPhotonicHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPhotonicHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "MolecularHardware" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MolecularHardware_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMolecularHardware[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMolecularHardware[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMolecularHardware {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "PowerDomain" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PowerDomain_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPowerDomain[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPowerDomain[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPowerDomain {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SpatialArray" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SpatialArray_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSpatialArray[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSpatialArray[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSpatialArray {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Kernel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Kernel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApKernel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApKernel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApKernel {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FusionPattern" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FusionPattern_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFusionPattern[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFusionPattern[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFusionPattern {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FusionHardwareTarget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FusionHardwareTarget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFusionHardwareTarget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFusionHardwareTarget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFusionHardwareTarget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FusionOpTarget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FusionOpTarget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFusionOpTarget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFusionOpTarget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFusionOpTarget {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "DataFormatConverter" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DataFormatConverter_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDataFormatConverter[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDataFormatConverter[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDataFormatConverter {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "OptimizationStrategy" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OptimizationStrategy_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOptimizationStrategy[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOptimizationStrategy[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOptimizationStrategy {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FitnessFunction" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FitnessFunction_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFitnessFunction[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFitnessFunction[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFitnessFunction {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FitnessComponent" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FitnessComponent_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFitnessComponent[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFitnessComponent[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFitnessComponent {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FaultModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FaultModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFaultModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFaultModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFaultModel {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Simulator" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Simulator_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSimulator[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSimulator[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSimulator {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "SimulationLevel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SimulationLevel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSimulationLevel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSimulationLevel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSimulationLevel {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "AdaptiveFidelity" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AdaptiveFidelity_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAdaptiveFidelity[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAdaptiveFidelity[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAdaptiveFidelity {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "DeterminismConstraint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DeterminismConstraint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDeterminismConstraint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDeterminismConstraint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDeterminismConstraint {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Constraint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Constraint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApConstraint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApConstraint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApConstraint {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Metric" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Metric_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMetric[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMetric[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMetric {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Checkpoint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Checkpoint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCheckpoint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCheckpoint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCheckpoint {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ParadigmRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ParadigmRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApParadigmRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApParadigmRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApParadigmRule {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "NeuronModelRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuronModelRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuronModelRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuronModelRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuronModelRule {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "QuantumGateRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["QuantumGateRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApQuantumGateRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApQuantumGateRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApQuantumGateRule {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "DataflowRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DataflowRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDataflowRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDataflowRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDataflowRule {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "FusionStrategyRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FusionStrategyRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFusionStrategyRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFusionStrategyRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFusionStrategyRule {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Extension" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Extension_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApExtension[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApExtension[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApExtension {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "CustomParadigm" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CustomParadigm_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCustomParadigm[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCustomParadigm[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCustomParadigm {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Actor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Actor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApActor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApActor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApActor {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	fmt.Printf("?No all %s cmd ?%s? > g_runh.act:43", va[0], lno);
	return 0;
}

func Load(act *ActT, toks string, ln string, pos int, lno string) int {
	errs := 0
	ss := strings.Split(toks,".")
	tok := ss[0]
	flag := ss[1:]
	if tok == "Actor" { errs += loadActor(act,ln,pos,lno,flag) }
	if tok == "All" { errs += loadAll(act,ln,pos,lno,flag) }
	if tok == "Du" { errs += loadDu(act,ln,pos,lno,flag) }
	if tok == "New" { errs += loadNew(act,ln,pos,lno,flag) }
	if tok == "Refs" { errs += loadRefs(act,ln,pos,lno,flag) }
	if tok == "Var" { errs += loadVar(act,ln,pos,lno,flag) }
	if tok == "Its" { errs += loadIts(act,ln,pos,lno,flag) }
	if tok == "C" { errs += loadC(act,ln,pos,lno,flag) }
	if tok == "Cs" { errs += loadCs(act,ln,pos,lno,flag) }
	if tok == "Out" { errs += loadOut(act,ln,pos,lno,flag) }
	if tok == "In" { errs += loadIn(act,ln,pos,lno,flag) }
	if tok == "Break" { errs += loadBreak(act,ln,pos,lno,flag) }
	if tok == "Add" { errs += loadAdd(act,ln,pos,lno,flag) }
	if tok == "This" { errs += loadThis(act,ln,pos,lno,flag) }
	if tok == "Replace" { errs += loadReplace(act,ln,pos,lno,flag) }
	return errs
}

func Loadh(act *ActT, toks string, ln string, pos int, lno string, nm map[string]string) int {
	errs := 0
	ss := strings.Split(toks,".")
	tok := ss[0]
	flag := ss[1:]
	if tok == "Project" { errs += loadProject(act,ln,pos,lno,flag,nm) }
	if tok == "ComputeGraph" { errs += loadComputeGraph(act,ln,pos,lno,flag,nm) }
	if tok == "HardwareTarget" { errs += loadHardwareTarget(act,ln,pos,lno,flag,nm) }
	if tok == "DataTensor" { errs += loadDataTensor(act,ln,pos,lno,flag,nm) }
	if tok == "TensorConsumer" { errs += loadTensorConsumer(act,ln,pos,lno,flag,nm) }
	if tok == "Operation" { errs += loadOperation(act,ln,pos,lno,flag,nm) }
	if tok == "OperationArg" { errs += loadOperationArg(act,ln,pos,lno,flag,nm) }
	if tok == "OpDependency" { errs += loadOpDependency(act,ln,pos,lno,flag,nm) }
	if tok == "ClassicalOp" { errs += loadClassicalOp(act,ln,pos,lno,flag,nm) }
	if tok == "SpikingOp" { errs += loadSpikingOp(act,ln,pos,lno,flag,nm) }
	if tok == "PlasticityRule" { errs += loadPlasticityRule(act,ln,pos,lno,flag,nm) }
	if tok == "AnalogOp" { errs += loadAnalogOp(act,ln,pos,lno,flag,nm) }
	if tok == "QuantumOp" { errs += loadQuantumOp(act,ln,pos,lno,flag,nm) }
	if tok == "QubitTarget" { errs += loadQubitTarget(act,ln,pos,lno,flag,nm) }
	if tok == "ControlQubit" { errs += loadControlQubit(act,ln,pos,lno,flag,nm) }
	if tok == "QuantumCircuit" { errs += loadQuantumCircuit(act,ln,pos,lno,flag,nm) }
	if tok == "PhotonicOp" { errs += loadPhotonicOp(act,ln,pos,lno,flag,nm) }
	if tok == "MolecularOp" { errs += loadMolecularOp(act,ln,pos,lno,flag,nm) }
	if tok == "Reactant" { errs += loadReactant(act,ln,pos,lno,flag,nm) }
	if tok == "Product" { errs += loadProduct(act,ln,pos,lno,flag,nm) }
	if tok == "HybridOp" { errs += loadHybridOp(act,ln,pos,lno,flag,nm) }
	if tok == "FallbackMode" { errs += loadFallbackMode(act,ln,pos,lno,flag,nm) }
	if tok == "RedundancyStrategy" { errs += loadRedundancyStrategy(act,ln,pos,lno,flag,nm) }
	if tok == "ProfilingHook" { errs += loadProfilingHook(act,ln,pos,lno,flag,nm) }
	if tok == "AdaptiveParameter" { errs += loadAdaptiveParameter(act,ln,pos,lno,flag,nm) }
	if tok == "SecureCompute" { errs += loadSecureCompute(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpace" { errs += loadSearchSpace(act,ln,pos,lno,flag,nm) }
	if tok == "SearchTarget" { errs += loadSearchTarget(act,ln,pos,lno,flag,nm) }
	if tok == "SearchParameter" { errs += loadSearchParameter(act,ln,pos,lno,flag,nm) }
	if tok == "EnergyBudget" { errs += loadEnergyBudget(act,ln,pos,lno,flag,nm) }
	if tok == "EnergyAllocation" { errs += loadEnergyAllocation(act,ln,pos,lno,flag,nm) }
	if tok == "TileMapping" { errs += loadTileMapping(act,ln,pos,lno,flag,nm) }
	if tok == "TileTarget" { errs += loadTileTarget(act,ln,pos,lno,flag,nm) }
	if tok == "PrivacyBudget" { errs += loadPrivacyBudget(act,ln,pos,lno,flag,nm) }
	if tok == "TraceCollection" { errs += loadTraceCollection(act,ln,pos,lno,flag,nm) }
	if tok == "EvolvableGraph" { errs += loadEvolvableGraph(act,ln,pos,lno,flag,nm) }
	if tok == "Provenance" { errs += loadProvenance(act,ln,pos,lno,flag,nm) }
	if tok == "OptimizationRun" { errs += loadOptimizationRun(act,ln,pos,lno,flag,nm) }
	if tok == "Validation" { errs += loadValidation(act,ln,pos,lno,flag,nm) }
	if tok == "Genotype" { errs += loadGenotype(act,ln,pos,lno,flag,nm) }
	if tok == "Phenotype" { errs += loadPhenotype(act,ln,pos,lno,flag,nm) }
	if tok == "Hardware" { errs += loadHardware(act,ln,pos,lno,flag,nm) }
	if tok == "PhysicsModel" { errs += loadPhysicsModel(act,ln,pos,lno,flag,nm) }
	if tok == "ClassicalHardware" { errs += loadClassicalHardware(act,ln,pos,lno,flag,nm) }
	if tok == "NeuromorphicHardware" { errs += loadNeuromorphicHardware(act,ln,pos,lno,flag,nm) }
	if tok == "AnalogHardware" { errs += loadAnalogHardware(act,ln,pos,lno,flag,nm) }
	if tok == "QuantumHardware" { errs += loadQuantumHardware(act,ln,pos,lno,flag,nm) }
	if tok == "PhotonicHardware" { errs += loadPhotonicHardware(act,ln,pos,lno,flag,nm) }
	if tok == "MolecularHardware" { errs += loadMolecularHardware(act,ln,pos,lno,flag,nm) }
	if tok == "PowerDomain" { errs += loadPowerDomain(act,ln,pos,lno,flag,nm) }
	if tok == "SpatialArray" { errs += loadSpatialArray(act,ln,pos,lno,flag,nm) }
	if tok == "Kernel" { errs += loadKernel(act,ln,pos,lno,flag,nm) }
	if tok == "FusionPattern" { errs += loadFusionPattern(act,ln,pos,lno,flag,nm) }
	if tok == "FusionHardwareTarget" { errs += loadFusionHardwareTarget(act,ln,pos,lno,flag,nm) }
	if tok == "FusionOpTarget" { errs += loadFusionOpTarget(act,ln,pos,lno,flag,nm) }
	if tok == "DataFormatConverter" { errs += loadDataFormatConverter(act,ln,pos,lno,flag,nm) }
	if tok == "OptimizationStrategy" { errs += loadOptimizationStrategy(act,ln,pos,lno,flag,nm) }
	if tok == "FitnessFunction" { errs += loadFitnessFunction(act,ln,pos,lno,flag,nm) }
	if tok == "FitnessComponent" { errs += loadFitnessComponent(act,ln,pos,lno,flag,nm) }
	if tok == "FaultModel" { errs += loadFaultModel(act,ln,pos,lno,flag,nm) }
	if tok == "Simulator" { errs += loadSimulator(act,ln,pos,lno,flag,nm) }
	if tok == "SimulationLevel" { errs += loadSimulationLevel(act,ln,pos,lno,flag,nm) }
	if tok == "AdaptiveFidelity" { errs += loadAdaptiveFidelity(act,ln,pos,lno,flag,nm) }
	if tok == "DeterminismConstraint" { errs += loadDeterminismConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "Constraint" { errs += loadConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "Metric" { errs += loadMetric(act,ln,pos,lno,flag,nm) }
	if tok == "Checkpoint" { errs += loadCheckpoint(act,ln,pos,lno,flag,nm) }
	if tok == "ParadigmRule" { errs += loadParadigmRule(act,ln,pos,lno,flag,nm) }
	if tok == "NeuronModelRule" { errs += loadNeuronModelRule(act,ln,pos,lno,flag,nm) }
	if tok == "QuantumGateRule" { errs += loadQuantumGateRule(act,ln,pos,lno,flag,nm) }
	if tok == "DataflowRule" { errs += loadDataflowRule(act,ln,pos,lno,flag,nm) }
	if tok == "FusionStrategyRule" { errs += loadFusionStrategyRule(act,ln,pos,lno,flag,nm) }
	if tok == "Extension" { errs += loadExtension(act,ln,pos,lno,flag,nm) }
	if tok == "CustomParadigm" { errs += loadCustomParadigm(act,ln,pos,lno,flag,nm) }
	return errs
}

