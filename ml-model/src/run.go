package main

import (
	"strings"
	"fmt"
	"strconv"
)

type ActT struct {
	index       map[string]int
	ApDomain [] *KpDomain
	ApKernel [] *KpKernel
	ApKernelParam [] *KpKernelParam
	ApKernelOp [] *KpKernelOp
	ApOptimization [] *KpOptimization
	ApScheduleOp [] *KpScheduleOp
	ApMemLayout [] *KpMemLayout
	ApFusion [] *KpFusion
	ApModel [] *KpModel
	ApBlock [] *KpBlock
	ApBlockParam [] *KpBlockParam
	ApBlockInstance [] *KpBlockInstance
	ApParamValue [] *KpParamValue
	ApTensor [] *KpTensor
	ApLayer [] *KpLayer
	ApOp [] *KpOp
	ApArg [] *KpArg
	ApAttentionOp [] *KpAttentionOp
	ApAttentionFreeOp [] *KpAttentionFreeOp
	ApRecurrentFormulation [] *KpRecurrentFormulation
	ApGraphOp [] *KpGraphOp
	ApGraphLearningOp [] *KpGraphLearningOp
	ApStatefulOp [] *KpStatefulOp
	ApLTCOp [] *KpLTCOp
	ApODEOp [] *KpODEOp
	ApSDEOp [] *KpSDEOp
	ApPDEOp [] *KpPDEOp
	ApMemoryOp [] *KpMemoryOp
	ApExpertRoutingOp [] *KpExpertRoutingOp
	ApCapacityAwareRoutingOp [] *KpCapacityAwareRoutingOp
	ApSamplingOp [] *KpSamplingOp
	ApStochasticOp [] *KpStochasticOp
	ApDynamicRoutingOp [] *KpDynamicRoutingOp
	ApDynamicOp [] *KpDynamicOp
	ApNeuralProgramOp [] *KpNeuralProgramOp
	ApWeightGenerationOp [] *KpWeightGenerationOp
	ApArchitectureSearchOp [] *KpArchitectureSearchOp
	ApContinuousDepthOp [] *KpContinuousDepthOp
	ApOpTypeRule [] *KpOpTypeRule
	ApArgRoleRule [] *KpArgRoleRule
	ApConfig [] *KpConfig
	ApSchedule [] *KpSchedule
	ApControlFlow [] *KpControlFlow
	ApCondition [] *KpCondition
	ApBranch [] *KpBranch
	ApGraphTensor [] *KpGraphTensor
	ApStateTransfer [] *KpStateTransfer
	ApContinuousLayer [] *KpContinuousLayer
	ApSearchSpace [] *KpSearchSpace
	ApSearchOp [] *KpSearchOp
	ApSearchEdge [] *KpSearchEdge
	ApArchitectureParam [] *KpArchitectureParam
	ApArchitectureGradient [] *KpArchitectureGradient
	ApHyperNetwork [] *KpHyperNetwork
	ApWeightGenerator [] *KpWeightGenerator
	ApMemoryMatrix [] *KpMemoryMatrix
	ApDifferentiableProgram [] *KpDifferentiableProgram
	ApProgramInstruction [] *KpProgramInstruction
	ApMetaLearner [] *KpMetaLearner
	ApSparseExpertSystem [] *KpSparseExpertSystem
	ApCapacityAwareRouter [] *KpCapacityAwareRouter
	ApNeuralSDE [] *KpNeuralSDE
	ApNeuralPDE [] *KpNeuralPDE
	ApDynamicGraphNetwork [] *KpDynamicGraphNetwork
	ApGraphLearner [] *KpGraphLearner
	ApNeuralProgram [] *KpNeuralProgram
	ApProgInstruction [] *KpProgInstruction
	ApLiquidNetwork [] *KpLiquidNetwork
	ApSymbolicShape [] *KpSymbolicShape
	ApJITCompiler [] *KpJITCompiler
	ApTargetRule [] *KpTargetRule
	ApDtypeRule [] *KpDtypeRule
	ApFlagRule [] *KpFlagRule
	ApDistributionRule [] *KpDistributionRule
	ApSearchStrategyRule [] *KpSearchStrategyRule
	ApDynamicGraphRule [] *KpDynamicGraphRule
	ApCompatibilityNote [] *KpCompatibilityNote
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
	for _, st := range act.ApOptimization {

//  net5.unit:57, g_runh.act:180

		v, _ = st.Names["flags"]
		err, res = fnd3(act, "FlagRule_" + v, v, "ref:Optimization.flags:FlagRule." + v,  "+", st.LineNo, "net5.unit:57, g_runh.act:184" );
		st.Kflagsp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApBlock {

//  net5.unit:115, g_runh.act:180

		v, _ = st.Names["model"]
		err, res = fnd3(act, "Model_" + v, v, "ref:Block.model:Model." + v,  "+", st.LineNo, "net5.unit:115, g_runh.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApBlockInstance {

//  net5.unit:136, g_runh.act:180

		v, _ = st.Names["block"]
		err, res = fnd3(act, "Block_" + v, v, "ref:BlockInstance.block:Block." + v,  "+", st.LineNo, "net5.unit:136, g_runh.act:184" );
		st.Kblockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTensor {

//  net5.unit:160, g_runh.act:180

		v, _ = st.Names["dtype"]
		err, res = fnd3(act, "DtypeRule_" + v, v, "ref:Tensor.dtype:DtypeRule." + v,  "+", st.LineNo, "net5.unit:160, g_runh.act:184" );
		st.Kdtypep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOp {

//  net5.unit:185, g_runh.act:194

		v, _ = st.Names["op_type"]
		for id, child := range st.Childs {
			if child.TypeName() == v {
				st.Kop_typep = id
				break
			}
		}
//  net5.unit:190, g_runh.act:180

		v, _ = st.Names["kernel"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:Op.kernel:Kernel." + v,  "*", st.LineNo, "net5.unit:190, g_runh.act:184" );
		st.Kkernelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApArgRoleRule {

//  net5.unit:629, g_runh.act:180

		v, _ = st.Names["dtype"]
		err, res = fnd3(act, "DtypeRule_" + v, v, "ref:ArgRoleRule.dtype:DtypeRule." + v,  "+", st.LineNo, "net5.unit:629, g_runh.act:184" );
		st.Kdtypep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApConfig {

//  net5.unit:639, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "TargetRule_" + v, v, "ref:Config.target:TargetRule." + v,  "+", st.LineNo, "net5.unit:639, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
//  net5.unit:641, g_runh.act:180

		v, _ = st.Names["opt_flags"]
		err, res = fnd3(act, "FlagRule_" + v, v, "ref:Config.opt_flags:FlagRule." + v,  "*", st.LineNo, "net5.unit:641, g_runh.act:184" );
		st.Kopt_flagsp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApGraphTensor {

//  net5.unit:711, g_runh.act:180

		v, _ = st.Names["dtype"]
		err, res = fnd3(act, "DtypeRule_" + v, v, "ref:GraphTensor.dtype:DtypeRule." + v,  "+", st.LineNo, "net5.unit:711, g_runh.act:184" );
		st.Kdtypep = res
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
	for _, st := range act.ApBlock {

//  net5.unit:116, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Block.layers unresolved from ref:Block.model:Model %s > net5.unit:116, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["layers"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Layer_" + v, v, "ref_child:Block.layers:Model." + parent + "." + v + " from ref:Block.model", "*", st.LineNo, "net5.unit:116, g_runh.act:236")
		st.Klayersp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:117, g_runh.act:224

 
	if st.Klayersp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Block.parameters unresolved from ref_child:Block.layers:Layer %s > net5.unit:117, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApLayer[st.Klayersp].MyName
		v, _ = st.Names["parameters"]
		err, res = fnd3(act, strconv.Itoa(st.Klayersp) + "_BlockParam_" + v, v, "ref_child:Block.parameters:Layer." + parent + "." + v + " from ref_child:Block.layers", "*", st.LineNo, "net5.unit:117, g_runh.act:236")
		st.Kparametersp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApBlockInstance {

//  net5.unit:137, g_runh.act:224

 
	if st.Kblockp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:BlockInstance.param_values unresolved from ref:BlockInstance.block:Block %s > net5.unit:137, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApBlock[st.Kblockp].MyName
		v, _ = st.Names["param_values"]
		err, res = fnd3(act, strconv.Itoa(st.Kblockp) + "_ParamValue_" + v, v, "ref_child:BlockInstance.param_values:Block." + parent + "." + v + " from ref:BlockInstance.block", "*", st.LineNo, "net5.unit:137, g_runh.act:236")
		st.Kparam_valuesp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApOp {

//  net5.unit:191, g_runh.act:224

 
	if st.Kkernelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Op.kernel_op unresolved from ref:Op.kernel:Kernel %s > net5.unit:191, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApKernel[st.Kkernelp].MyName
		v, _ = st.Names["kernel_op"]
		err, res = fnd3(act, strconv.Itoa(st.Kkernelp) + "_KernelOp_" + v, v, "ref_child:Op.kernel_op:Kernel." + parent + "." + v + " from ref:Op.kernel", "*", st.LineNo, "net5.unit:191, g_runh.act:236")
		st.Kkernel_opp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApArg {

//  net5.unit:202, g_runh.act:262
	p = st.Kparentp
	p = act.ApOp[p].Kparentp
	p = act.ApLayer[p].Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:Arg.model unresolved from word:Arg.role:..x %s (+) > net5.unit:202, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:203, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:Arg.tensor unresolved from up_copy:Arg.model:Model %s > net5.unit:203, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["tensor"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Tensor_" + v, v, "ref_child:Arg.tensor:Model." + parent + "." + v + " from up_copy:Arg.model", "+", st.LineNo, "net5.unit:203, g_runh.act:236")
		st.Ktensorp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApExpertRoutingOp {

//  net5.unit:375, g_runh.act:262
	p = st.Kparentp
	p = act.ApOp[p].Kparentp
	p = act.ApLayer[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ExpertRoutingOp.model2 unresolved from word:ExpertRoutingOp.top_k:..x %s (+) > net5.unit:375, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:376, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:ExpertRoutingOp.experts unresolved from up_copy:ExpertRoutingOp.model2:Model %s > net5.unit:376, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["experts"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Layer_" + v, v, "ref_child:ExpertRoutingOp.experts:Model." + parent + "." + v + " from up_copy:ExpertRoutingOp.model2", "+", st.LineNo, "net5.unit:376, g_runh.act:236")
		st.Kexpertsp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSchedule {

//  net5.unit:650, g_runh.act:262
	p = st.Kparentp
	p = act.ApConfig[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:Schedule.model1 unresolved from word:Schedule.seq:..x %s (+) > net5.unit:650, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:651, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:Schedule.layer unresolved from up_copy:Schedule.model1:Model %s > net5.unit:651, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["layer"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Layer_" + v, v, "ref_child:Schedule.layer:Model." + parent + "." + v + " from up_copy:Schedule.model1", "+", st.LineNo, "net5.unit:651, g_runh.act:236")
		st.Klayerp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:652, g_runh.act:224

 
	if st.Klayerp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:Schedule.op unresolved from ref_child:Schedule.layer:Layer %s > net5.unit:652, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApLayer[st.Klayerp].MyName
		v, _ = st.Names["op"]
		err, res = fnd3(act, strconv.Itoa(st.Klayerp) + "_Op_" + v, v, "ref_child:Schedule.op:Layer." + parent + "." + v + " from ref_child:Schedule.layer", "+", st.LineNo, "net5.unit:652, g_runh.act:236")
		st.Kopp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:653, g_runh.act:262
	p = st.Kparentp
	p = act.ApConfig[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:Schedule.model2 unresolved from ref_child:Schedule.op:Op.x %s (+) > net5.unit:653, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:654, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Schedule.control unresolved from up_copy:Schedule.model2:Model %s > net5.unit:654, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["control"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_ControlFlow_" + v, v, "ref_child:Schedule.control:Model." + parent + "." + v + " from up_copy:Schedule.model2", "*", st.LineNo, "net5.unit:654, g_runh.act:236")
		st.Kcontrolp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApCondition {

//  net5.unit:680, g_runh.act:262
	p = st.Kparentp
	p = act.ApControlFlow[p].Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:Condition.model unresolved from word:Condition.predicate:..x %s (+) > net5.unit:680, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:681, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Condition.input unresolved from up_copy:Condition.model:Model %s > net5.unit:681, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["input"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Tensor_" + v, v, "ref_child:Condition.input:Model." + parent + "." + v + " from up_copy:Condition.model", "*", st.LineNo, "net5.unit:681, g_runh.act:236")
		st.Kinputp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApBranch {

//  net5.unit:692, g_runh.act:262
	p = st.Kparentp
	p = act.ApControlFlow[p].Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:Branch.model unresolved from word:Branch.branch_id:..x %s (+) > net5.unit:692, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:693, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:Branch.layers unresolved from up_copy:Branch.model:Model %s > net5.unit:693, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["layers"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Layer_" + v, v, "ref_child:Branch.layers:Model." + parent + "." + v + " from up_copy:Branch.model", "+", st.LineNo, "net5.unit:693, g_runh.act:236")
		st.Klayersp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApStateTransfer {

//  net5.unit:722, g_runh.act:262
	p = st.Kparentp
	p = act.ApConfig[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:StateTransfer.model1 unresolved from word:StateTransfer.policy:..x %s (+) > net5.unit:722, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:723, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:StateTransfer.source_state unresolved from up_copy:StateTransfer.model1:Model %s > net5.unit:723, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["source_state"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Tensor_" + v, v, "ref_child:StateTransfer.source_state:Model." + parent + "." + v + " from up_copy:StateTransfer.model1", "+", st.LineNo, "net5.unit:723, g_runh.act:236")
		st.Ksource_statep = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:724, g_runh.act:262
	p = st.Kparentp
	p = act.ApConfig[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:StateTransfer.model2 unresolved from ref_child:StateTransfer.source_state:Tensor.x %s (+) > net5.unit:724, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:725, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:StateTransfer.target_state unresolved from up_copy:StateTransfer.model2:Model %s > net5.unit:725, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["target_state"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Tensor_" + v, v, "ref_child:StateTransfer.target_state:Model." + parent + "." + v + " from up_copy:StateTransfer.model2", "+", st.LineNo, "net5.unit:725, g_runh.act:236")
		st.Ktarget_statep = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApContinuousLayer {

//  net5.unit:743, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ContinuousLayer.model unresolved from word:ContinuousLayer.adaptive:..x %s (+) > net5.unit:743, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:744, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:ContinuousLayer.dynamics unresolved from up_copy:ContinuousLayer.model:Model %s > net5.unit:744, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["dynamics"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Layer_" + v, v, "ref_child:ContinuousLayer.dynamics:Model." + parent + "." + v + " from up_copy:ContinuousLayer.model", "+", st.LineNo, "net5.unit:744, g_runh.act:236")
		st.Kdynamicsp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSearchOp {

//  net5.unit:769, g_runh.act:262
	p = st.Kparentp
	p = act.ApSearchSpace[p].Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:SearchOp.model unresolved from key:SearchOp.search_op:..x %s (+) > net5.unit:769, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:770, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:SearchOp.layer unresolved from up_copy:SearchOp.model:Model %s > net5.unit:770, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["layer"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Layer_" + v, v, "ref_child:SearchOp.layer:Model." + parent + "." + v + " from up_copy:SearchOp.model", "+", st.LineNo, "net5.unit:770, g_runh.act:236")
		st.Klayerp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:771, g_runh.act:224

 
	if st.Klayerp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:SearchOp.operation unresolved from ref_child:SearchOp.layer:Layer %s > net5.unit:771, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApLayer[st.Klayerp].MyName
		v, _ = st.Names["operation"]
		err, res = fnd3(act, strconv.Itoa(st.Klayerp) + "_Op_" + v, v, "ref_child:SearchOp.operation:Layer." + parent + "." + v + " from ref_child:SearchOp.layer", "*", st.LineNo, "net5.unit:771, g_runh.act:236")
		st.Koperationp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApArchitectureGradient {

//  net5.unit:804, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ArchitectureGradient.model unresolved from key:ArchitectureGradient.arch_gradient:..x %s (+) > net5.unit:804, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:805, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ArchitectureGradient.arch_params unresolved from up_copy:ArchitectureGradient.model:Model %s > net5.unit:805, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["arch_params"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_ArchitectureParam_" + v, v, "ref_child:ArchitectureGradient.arch_params:Model." + parent + "." + v + " from up_copy:ArchitectureGradient.model", "*", st.LineNo, "net5.unit:805, g_runh.act:236")
		st.Karch_paramsp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:806, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ArchitectureGradient.model2 unresolved from ref_child:ArchitectureGradient.arch_params:ArchitectureParam.x %s (+) > net5.unit:806, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:807, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ArchitectureGradient.layer unresolved from up_copy:ArchitectureGradient.model2:Model %s > net5.unit:807, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["layer"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Layer_" + v, v, "ref_child:ArchitectureGradient.layer:Model." + parent + "." + v + " from up_copy:ArchitectureGradient.model2", "*", st.LineNo, "net5.unit:807, g_runh.act:236")
		st.Klayerp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:808, g_runh.act:224

 
	if st.Klayerp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ArchitectureGradient.gradient_ops unresolved from ref_child:ArchitectureGradient.layer:Layer %s > net5.unit:808, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApLayer[st.Klayerp].MyName
		v, _ = st.Names["gradient_ops"]
		err, res = fnd3(act, strconv.Itoa(st.Klayerp) + "_Op_" + v, v, "ref_child:ArchitectureGradient.gradient_ops:Layer." + parent + "." + v + " from ref_child:ArchitectureGradient.layer", "*", st.LineNo, "net5.unit:808, g_runh.act:236")
		st.Kgradient_opsp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApHyperNetwork {

//  net5.unit:823, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:HyperNetwork.model1 unresolved from key:HyperNetwork.hypernet:..x %s (+) > net5.unit:823, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:824, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:HyperNetwork.target_net unresolved from up_copy:HyperNetwork.model1:Model %s > net5.unit:824, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["target_net"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Layer_" + v, v, "ref_child:HyperNetwork.target_net:Model." + parent + "." + v + " from up_copy:HyperNetwork.model1", "*", st.LineNo, "net5.unit:824, g_runh.act:236")
		st.Ktarget_netp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:827, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:HyperNetwork.model2 unresolved from word:HyperNetwork.output_size:..x %s (+) > net5.unit:827, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:828, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:HyperNetwork.condition unresolved from up_copy:HyperNetwork.model2:Model %s > net5.unit:828, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["condition"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Tensor_" + v, v, "ref_child:HyperNetwork.condition:Model." + parent + "." + v + " from up_copy:HyperNetwork.model2", "*", st.LineNo, "net5.unit:828, g_runh.act:236")
		st.Kconditionp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApWeightGenerator {

//  net5.unit:838, g_runh.act:262
	p = st.Kparentp
	p = act.ApHyperNetwork[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:WeightGenerator.model1 unresolved from word:WeightGenerator.target_param:..x %s (+) > net5.unit:838, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:839, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:WeightGenerator.input_tensor unresolved from up_copy:WeightGenerator.model1:Model %s > net5.unit:839, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["input_tensor"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Tensor_" + v, v, "ref_child:WeightGenerator.input_tensor:Model." + parent + "." + v + " from up_copy:WeightGenerator.model1", "*", st.LineNo, "net5.unit:839, g_runh.act:236")
		st.Kinput_tensorp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:840, g_runh.act:262
	p = st.Kparentp
	p = act.ApHyperNetwork[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:WeightGenerator.model2 unresolved from ref_child:WeightGenerator.input_tensor:Tensor.x %s (+) > net5.unit:840, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:841, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:WeightGenerator.output_tensor unresolved from up_copy:WeightGenerator.model2:Model %s > net5.unit:841, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["output_tensor"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Tensor_" + v, v, "ref_child:WeightGenerator.output_tensor:Model." + parent + "." + v + " from up_copy:WeightGenerator.model2", "*", st.LineNo, "net5.unit:841, g_runh.act:236")
		st.Koutput_tensorp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApProgramInstruction {

//  net5.unit:883, g_runh.act:262
	p = st.Kparentp
	p = act.ApDifferentiableProgram[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ProgramInstruction.model1 unresolved from word:ProgramInstruction.op_type:..x %s (+) > net5.unit:883, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:884, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ProgramInstruction.operands unresolved from up_copy:ProgramInstruction.model1:Model %s > net5.unit:884, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["operands"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Tensor_" + v, v, "ref_child:ProgramInstruction.operands:Model." + parent + "." + v + " from up_copy:ProgramInstruction.model1", "*", st.LineNo, "net5.unit:884, g_runh.act:236")
		st.Koperandsp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:885, g_runh.act:262
	p = st.Kparentp
	p = act.ApDifferentiableProgram[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ProgramInstruction.model2 unresolved from ref_child:ProgramInstruction.operands:Tensor.x %s (+) > net5.unit:885, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:886, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ProgramInstruction.control_flow unresolved from up_copy:ProgramInstruction.model2:Model %s > net5.unit:886, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["control_flow"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_ControlFlow_" + v, v, "ref_child:ProgramInstruction.control_flow:Model." + parent + "." + v + " from up_copy:ProgramInstruction.model2", "*", st.LineNo, "net5.unit:886, g_runh.act:236")
		st.Kcontrol_flowp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:887, g_runh.act:224

 
	if st.Kcontrol_flowp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ProgramInstruction.condition unresolved from ref_child:ProgramInstruction.control_flow:ControlFlow %s > net5.unit:887, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApControlFlow[st.Kcontrol_flowp].MyName
		v, _ = st.Names["condition"]
		err, res = fnd3(act, strconv.Itoa(st.Kcontrol_flowp) + "_Condition_" + v, v, "ref_child:ProgramInstruction.condition:ControlFlow." + parent + "." + v + " from ref_child:ProgramInstruction.control_flow", "*", st.LineNo, "net5.unit:887, g_runh.act:236")
		st.Kconditionp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApMetaLearner {

//  net5.unit:897, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:MetaLearner.model1 unresolved from key:MetaLearner.meta_learner:..x %s (+) > net5.unit:897, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:898, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:MetaLearner.inner_loop unresolved from up_copy:MetaLearner.model1:Model %s > net5.unit:898, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["inner_loop"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Layer_" + v, v, "ref_child:MetaLearner.inner_loop:Model." + parent + "." + v + " from up_copy:MetaLearner.model1", "*", st.LineNo, "net5.unit:898, g_runh.act:236")
		st.Kinner_loopp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:899, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:MetaLearner.model2 unresolved from ref_child:MetaLearner.inner_loop:Layer.x %s (+) > net5.unit:899, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:900, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:MetaLearner.outer_loop unresolved from up_copy:MetaLearner.model2:Model %s > net5.unit:900, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["outer_loop"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Layer_" + v, v, "ref_child:MetaLearner.outer_loop:Model." + parent + "." + v + " from up_copy:MetaLearner.model2", "*", st.LineNo, "net5.unit:900, g_runh.act:236")
		st.Kouter_loopp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSparseExpertSystem {

//  net5.unit:915, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:SparseExpertSystem.model unresolved from key:SparseExpertSystem.expert_system:..x %s (+) > net5.unit:915, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:916, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:SparseExpertSystem.experts unresolved from up_copy:SparseExpertSystem.model:Model %s > net5.unit:916, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["experts"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Layer_" + v, v, "ref_child:SparseExpertSystem.experts:Model." + parent + "." + v + " from up_copy:SparseExpertSystem.model", "*", st.LineNo, "net5.unit:916, g_runh.act:236")
		st.Kexpertsp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApCapacityAwareRouter {

//  net5.unit:929, g_runh.act:262
	p = st.Kparentp
	p = act.ApSparseExpertSystem[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:CapacityAwareRouter.model1 unresolved from key:CapacityAwareRouter.router:..x %s (+) > net5.unit:929, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:930, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:CapacityAwareRouter.gate_input unresolved from up_copy:CapacityAwareRouter.model1:Model %s > net5.unit:930, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["gate_input"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Tensor_" + v, v, "ref_child:CapacityAwareRouter.gate_input:Model." + parent + "." + v + " from up_copy:CapacityAwareRouter.model1", "+", st.LineNo, "net5.unit:930, g_runh.act:236")
		st.Kgate_inputp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:931, g_runh.act:262
	p = st.Kparentp
	p = act.ApSparseExpertSystem[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:CapacityAwareRouter.model2 unresolved from ref_child:CapacityAwareRouter.gate_input:Tensor.x %s (+) > net5.unit:931, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:932, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:CapacityAwareRouter.expert_mask unresolved from up_copy:CapacityAwareRouter.model2:Model %s > net5.unit:932, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["expert_mask"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Tensor_" + v, v, "ref_child:CapacityAwareRouter.expert_mask:Model." + parent + "." + v + " from up_copy:CapacityAwareRouter.model2", "*", st.LineNo, "net5.unit:932, g_runh.act:236")
		st.Kexpert_maskp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:933, g_runh.act:262
	p = st.Kparentp
	p = act.ApSparseExpertSystem[p].Kparentp
	if p >= 0 {
		st.Kmodel3p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:CapacityAwareRouter.model3 unresolved from ref_child:CapacityAwareRouter.expert_mask:Tensor.x %s (+) > net5.unit:933, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:934, g_runh.act:224

 
	if st.Kmodel3p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:CapacityAwareRouter.token_assign unresolved from up_copy:CapacityAwareRouter.model3:Model %s > net5.unit:934, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel3p].MyName
		v, _ = st.Names["token_assign"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel3p) + "_Tensor_" + v, v, "ref_child:CapacityAwareRouter.token_assign:Model." + parent + "." + v + " from up_copy:CapacityAwareRouter.model3", "*", st.LineNo, "net5.unit:934, g_runh.act:236")
		st.Ktoken_assignp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApNeuralSDE {

//  net5.unit:948, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:NeuralSDE.model1 unresolved from key:NeuralSDE.neural_sde:..x %s (+) > net5.unit:948, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:949, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:NeuralSDE.drift_net unresolved from up_copy:NeuralSDE.model1:Model %s > net5.unit:949, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["drift_net"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Layer_" + v, v, "ref_child:NeuralSDE.drift_net:Model." + parent + "." + v + " from up_copy:NeuralSDE.model1", "+", st.LineNo, "net5.unit:949, g_runh.act:236")
		st.Kdrift_netp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:950, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:NeuralSDE.model2 unresolved from ref_child:NeuralSDE.drift_net:Layer.x %s (+) > net5.unit:950, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:951, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:NeuralSDE.diffusion_net unresolved from up_copy:NeuralSDE.model2:Model %s > net5.unit:951, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["diffusion_net"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Layer_" + v, v, "ref_child:NeuralSDE.diffusion_net:Model." + parent + "." + v + " from up_copy:NeuralSDE.model2", "+", st.LineNo, "net5.unit:951, g_runh.act:236")
		st.Kdiffusion_netp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApNeuralPDE {

//  net5.unit:963, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:NeuralPDE.model1 unresolved from word:NeuralPDE.pde_type:..x %s (+) > net5.unit:963, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:964, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:NeuralPDE.domain unresolved from up_copy:NeuralPDE.model1:Model %s > net5.unit:964, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["domain"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Tensor_" + v, v, "ref_child:NeuralPDE.domain:Model." + parent + "." + v + " from up_copy:NeuralPDE.model1", "+", st.LineNo, "net5.unit:964, g_runh.act:236")
		st.Kdomainp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:965, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:NeuralPDE.model2 unresolved from ref_child:NeuralPDE.domain:Tensor.x %s (+) > net5.unit:965, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:966, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:NeuralPDE.boundary unresolved from up_copy:NeuralPDE.model2:Model %s > net5.unit:966, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["boundary"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Tensor_" + v, v, "ref_child:NeuralPDE.boundary:Model." + parent + "." + v + " from up_copy:NeuralPDE.model2", "*", st.LineNo, "net5.unit:966, g_runh.act:236")
		st.Kboundaryp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:967, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel3p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:NeuralPDE.model3 unresolved from ref_child:NeuralPDE.boundary:Tensor.x %s (+) > net5.unit:967, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:968, g_runh.act:224

 
	if st.Kmodel3p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:NeuralPDE.initial unresolved from up_copy:NeuralPDE.model3:Model %s > net5.unit:968, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel3p].MyName
		v, _ = st.Names["initial"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel3p) + "_Tensor_" + v, v, "ref_child:NeuralPDE.initial:Model." + parent + "." + v + " from up_copy:NeuralPDE.model3", "*", st.LineNo, "net5.unit:968, g_runh.act:236")
		st.Kinitialp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApDynamicGraphNetwork {

//  net5.unit:982, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:DynamicGraphNetwork.model1 unresolved from key:DynamicGraphNetwork.dynamic_graph:..x %s (+) > net5.unit:982, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:983, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:DynamicGraphNetwork.node_features unresolved from up_copy:DynamicGraphNetwork.model1:Model %s > net5.unit:983, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["node_features"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_GraphTensor_" + v, v, "ref_child:DynamicGraphNetwork.node_features:Model." + parent + "." + v + " from up_copy:DynamicGraphNetwork.model1", "+", st.LineNo, "net5.unit:983, g_runh.act:236")
		st.Knode_featuresp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:984, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:DynamicGraphNetwork.model2 unresolved from ref_child:DynamicGraphNetwork.node_features:GraphTensor.x %s (+) > net5.unit:984, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:985, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:DynamicGraphNetwork.edge_predictor unresolved from up_copy:DynamicGraphNetwork.model2:Model %s > net5.unit:985, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["edge_predictor"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_Layer_" + v, v, "ref_child:DynamicGraphNetwork.edge_predictor:Model." + parent + "." + v + " from up_copy:DynamicGraphNetwork.model2", "+", st.LineNo, "net5.unit:985, g_runh.act:236")
		st.Kedge_predictorp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApGraphLearner {

//  net5.unit:997, g_runh.act:262
	p = st.Kparentp
	p = act.ApDynamicGraphNetwork[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:GraphLearner.model1 unresolved from key:GraphLearner.learner:..x %s (+) > net5.unit:997, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:998, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:GraphLearner.input_nodes unresolved from up_copy:GraphLearner.model1:Model %s > net5.unit:998, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["input_nodes"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_GraphTensor_" + v, v, "ref_child:GraphLearner.input_nodes:Model." + parent + "." + v + " from up_copy:GraphLearner.model1", "+", st.LineNo, "net5.unit:998, g_runh.act:236")
		st.Kinput_nodesp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:999, g_runh.act:262
	p = st.Kparentp
	p = act.ApDynamicGraphNetwork[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:GraphLearner.model2 unresolved from ref_child:GraphLearner.input_nodes:GraphTensor.x %s (+) > net5.unit:999, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:1000, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:GraphLearner.adjacency unresolved from up_copy:GraphLearner.model2:Model %s > net5.unit:1000, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["adjacency"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_GraphTensor_" + v, v, "ref_child:GraphLearner.adjacency:Model." + parent + "." + v + " from up_copy:GraphLearner.model2", "+", st.LineNo, "net5.unit:1000, g_runh.act:236")
		st.Kadjacencyp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApNeuralProgram {

//  net5.unit:1017, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:NeuralProgram.model unresolved from word:NeuralProgram.program_counter:..x %s (+) > net5.unit:1017, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:1018, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:NeuralProgram.memory unresolved from up_copy:NeuralProgram.model:Model %s > net5.unit:1018, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["memory"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Tensor_" + v, v, "ref_child:NeuralProgram.memory:Model." + parent + "." + v + " from up_copy:NeuralProgram.model", "*", st.LineNo, "net5.unit:1018, g_runh.act:236")
		st.Kmemoryp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApProgInstruction {

//  net5.unit:1029, g_runh.act:262
	p = st.Kparentp
	p = act.ApNeuralProgram[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ProgInstruction.model1 unresolved from word:ProgInstruction.opcode:..x %s (+) > net5.unit:1029, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:1030, g_runh.act:224

 
	if st.Kmodel1p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ProgInstruction.operands unresolved from up_copy:ProgInstruction.model1:Model %s > net5.unit:1030, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel1p].MyName
		v, _ = st.Names["operands"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel1p) + "_Tensor_" + v, v, "ref_child:ProgInstruction.operands:Model." + parent + "." + v + " from up_copy:ProgInstruction.model1", "*", st.LineNo, "net5.unit:1030, g_runh.act:236")
		st.Koperandsp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:1031, g_runh.act:262
	p = st.Kparentp
	p = act.ApNeuralProgram[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:ProgInstruction.model2 unresolved from ref_child:ProgInstruction.operands:Tensor.x %s (+) > net5.unit:1031, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:1032, g_runh.act:224

 
	if st.Kmodel2p < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ProgInstruction.control_flow unresolved from up_copy:ProgInstruction.model2:Model %s > net5.unit:1032, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodel2p].MyName
		v, _ = st.Names["control_flow"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodel2p) + "_ControlFlow_" + v, v, "ref_child:ProgInstruction.control_flow:Model." + parent + "." + v + " from up_copy:ProgInstruction.model2", "*", st.LineNo, "net5.unit:1032, g_runh.act:236")
		st.Kcontrol_flowp = res
		if !err {
			errs += 1
		}
	}
//  net5.unit:1033, g_runh.act:224

 
	if st.Kcontrol_flowp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:ProgInstruction.condition unresolved from ref_child:ProgInstruction.control_flow:ControlFlow %s > net5.unit:1033, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApControlFlow[st.Kcontrol_flowp].MyName
		v, _ = st.Names["condition"]
		err, res = fnd3(act, strconv.Itoa(st.Kcontrol_flowp) + "_Condition_" + v, v, "ref_child:ProgInstruction.condition:ControlFlow." + parent + "." + v + " from ref_child:ProgInstruction.control_flow", "*", st.LineNo, "net5.unit:1033, g_runh.act:236")
		st.Kconditionp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApLiquidNetwork {

//  net5.unit:1047, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:LiquidNetwork.model unresolved from key:LiquidNetwork.liquid_net:..x %s (+) > net5.unit:1047, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  net5.unit:1048, g_runh.act:224

 
	if st.Kmodelp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:LiquidNetwork.time_const unresolved from up_copy:LiquidNetwork.model:Model %s > net5.unit:1048, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["time_const"]
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Tensor_" + v, v, "ref_child:LiquidNetwork.time_const:Model." + parent + "." + v + " from up_copy:LiquidNetwork.model", "+", st.LineNo, "net5.unit:1048, g_runh.act:236")
		st.Ktime_constp = res
		if !err {
			errs += 1
		}
	}
	}
	return(errs)
}

func DoAll(glob *GlobT, va []string, lno string) int {
	if va[0] == "Domain" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Domain_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDomain[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDomain[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDomain {
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
	if va[0] == "KernelParam" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["KernelParam_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApKernelParam[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApKernelParam[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApKernelParam {
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
	if va[0] == "KernelOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["KernelOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApKernelOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApKernelOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApKernelOp {
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
	if va[0] == "Optimization" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Optimization_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOptimization[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOptimization[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOptimization {
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
	if va[0] == "ScheduleOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ScheduleOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApScheduleOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApScheduleOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApScheduleOp {
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
	if va[0] == "MemLayout" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MemLayout_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMemLayout[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMemLayout[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMemLayout {
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
	if va[0] == "Fusion" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Fusion_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFusion[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFusion[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFusion {
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
	if va[0] == "Model" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Model_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApModel {
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
	if va[0] == "Block" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Block_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBlock[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBlock[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBlock {
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
	if va[0] == "BlockParam" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["BlockParam_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBlockParam[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBlockParam[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBlockParam {
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
	if va[0] == "BlockInstance" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["BlockInstance_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBlockInstance[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBlockInstance[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBlockInstance {
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
	if va[0] == "ParamValue" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ParamValue_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApParamValue[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApParamValue[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApParamValue {
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
	if va[0] == "Tensor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Tensor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTensor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTensor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTensor {
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
	if va[0] == "Layer" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Layer_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApLayer[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApLayer[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApLayer {
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
	if va[0] == "Op" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Op_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOp {
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
	if va[0] == "Arg" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Arg_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApArg[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApArg[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApArg {
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
	if va[0] == "AttentionOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AttentionOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAttentionOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAttentionOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAttentionOp {
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
	if va[0] == "AttentionFreeOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AttentionFreeOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAttentionFreeOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAttentionFreeOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAttentionFreeOp {
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
	if va[0] == "RecurrentFormulation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["RecurrentFormulation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApRecurrentFormulation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApRecurrentFormulation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApRecurrentFormulation {
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
	if va[0] == "GraphOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["GraphOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApGraphOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApGraphOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApGraphOp {
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
	if va[0] == "GraphLearningOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["GraphLearningOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApGraphLearningOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApGraphLearningOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApGraphLearningOp {
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
	if va[0] == "StatefulOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["StatefulOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApStatefulOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApStatefulOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApStatefulOp {
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
	if va[0] == "LTCOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["LTCOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApLTCOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApLTCOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApLTCOp {
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
	if va[0] == "ODEOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ODEOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApODEOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApODEOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApODEOp {
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
	if va[0] == "SDEOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SDEOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSDEOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSDEOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSDEOp {
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
	if va[0] == "PDEOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PDEOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPDEOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPDEOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPDEOp {
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
	if va[0] == "MemoryOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MemoryOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMemoryOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMemoryOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMemoryOp {
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
	if va[0] == "ExpertRoutingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ExpertRoutingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApExpertRoutingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApExpertRoutingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApExpertRoutingOp {
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
	if va[0] == "CapacityAwareRoutingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CapacityAwareRoutingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCapacityAwareRoutingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCapacityAwareRoutingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCapacityAwareRoutingOp {
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
	if va[0] == "SamplingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SamplingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSamplingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSamplingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSamplingOp {
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
	if va[0] == "StochasticOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["StochasticOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApStochasticOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApStochasticOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApStochasticOp {
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
	if va[0] == "DynamicRoutingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DynamicRoutingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDynamicRoutingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDynamicRoutingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDynamicRoutingOp {
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
	if va[0] == "DynamicOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DynamicOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDynamicOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDynamicOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDynamicOp {
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
	if va[0] == "NeuralProgramOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuralProgramOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuralProgramOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuralProgramOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuralProgramOp {
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
	if va[0] == "WeightGenerationOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["WeightGenerationOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApWeightGenerationOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApWeightGenerationOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApWeightGenerationOp {
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
	if va[0] == "ArchitectureSearchOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ArchitectureSearchOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApArchitectureSearchOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApArchitectureSearchOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApArchitectureSearchOp {
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
	if va[0] == "ContinuousDepthOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ContinuousDepthOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApContinuousDepthOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApContinuousDepthOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApContinuousDepthOp {
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
	if va[0] == "OpTypeRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OpTypeRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOpTypeRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOpTypeRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOpTypeRule {
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
	if va[0] == "ArgRoleRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ArgRoleRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApArgRoleRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApArgRoleRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApArgRoleRule {
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
	if va[0] == "Config" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Config_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApConfig[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApConfig[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApConfig {
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
	if va[0] == "Schedule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Schedule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSchedule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSchedule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSchedule {
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
	if va[0] == "ControlFlow" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ControlFlow_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApControlFlow[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApControlFlow[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApControlFlow {
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
	if va[0] == "Condition" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Condition_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCondition[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCondition[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCondition {
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
	if va[0] == "Branch" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Branch_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBranch[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBranch[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBranch {
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
	if va[0] == "GraphTensor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["GraphTensor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApGraphTensor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApGraphTensor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApGraphTensor {
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
	if va[0] == "StateTransfer" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["StateTransfer_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApStateTransfer[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApStateTransfer[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApStateTransfer {
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
	if va[0] == "ContinuousLayer" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ContinuousLayer_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApContinuousLayer[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApContinuousLayer[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApContinuousLayer {
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
	if va[0] == "SearchOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchOp {
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
	if va[0] == "SearchEdge" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchEdge_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchEdge[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchEdge[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchEdge {
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
	if va[0] == "ArchitectureParam" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ArchitectureParam_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApArchitectureParam[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApArchitectureParam[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApArchitectureParam {
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
	if va[0] == "ArchitectureGradient" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ArchitectureGradient_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApArchitectureGradient[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApArchitectureGradient[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApArchitectureGradient {
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
	if va[0] == "HyperNetwork" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HyperNetwork_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHyperNetwork[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHyperNetwork[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHyperNetwork {
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
	if va[0] == "WeightGenerator" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["WeightGenerator_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApWeightGenerator[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApWeightGenerator[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApWeightGenerator {
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
	if va[0] == "MemoryMatrix" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MemoryMatrix_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMemoryMatrix[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMemoryMatrix[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMemoryMatrix {
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
	if va[0] == "DifferentiableProgram" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DifferentiableProgram_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDifferentiableProgram[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDifferentiableProgram[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDifferentiableProgram {
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
	if va[0] == "ProgramInstruction" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ProgramInstruction_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProgramInstruction[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProgramInstruction[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProgramInstruction {
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
	if va[0] == "MetaLearner" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MetaLearner_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMetaLearner[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMetaLearner[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMetaLearner {
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
	if va[0] == "SparseExpertSystem" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SparseExpertSystem_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSparseExpertSystem[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSparseExpertSystem[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSparseExpertSystem {
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
	if va[0] == "CapacityAwareRouter" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CapacityAwareRouter_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCapacityAwareRouter[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCapacityAwareRouter[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCapacityAwareRouter {
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
	if va[0] == "NeuralSDE" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuralSDE_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuralSDE[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuralSDE[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuralSDE {
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
	if va[0] == "NeuralPDE" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuralPDE_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuralPDE[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuralPDE[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuralPDE {
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
	if va[0] == "DynamicGraphNetwork" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DynamicGraphNetwork_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDynamicGraphNetwork[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDynamicGraphNetwork[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
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
	if va[0] == "GraphLearner" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["GraphLearner_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApGraphLearner[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApGraphLearner[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApGraphLearner {
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
	if va[0] == "NeuralProgram" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuralProgram_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuralProgram[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuralProgram[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuralProgram {
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
	if va[0] == "ProgInstruction" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ProgInstruction_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProgInstruction[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProgInstruction[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProgInstruction {
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
	if va[0] == "LiquidNetwork" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["LiquidNetwork_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApLiquidNetwork[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApLiquidNetwork[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApLiquidNetwork {
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
	if va[0] == "SymbolicShape" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SymbolicShape_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSymbolicShape[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSymbolicShape[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSymbolicShape {
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
	if va[0] == "JITCompiler" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["JITCompiler_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApJITCompiler[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApJITCompiler[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApJITCompiler {
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
	if va[0] == "TargetRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TargetRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTargetRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTargetRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTargetRule {
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
	if va[0] == "DtypeRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DtypeRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDtypeRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDtypeRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDtypeRule {
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
	if va[0] == "FlagRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FlagRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFlagRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFlagRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFlagRule {
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
	if va[0] == "DistributionRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DistributionRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDistributionRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDistributionRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDistributionRule {
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
	if va[0] == "SearchStrategyRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchStrategyRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchStrategyRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchStrategyRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchStrategyRule {
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
	if va[0] == "DynamicGraphRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DynamicGraphRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDynamicGraphRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDynamicGraphRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDynamicGraphRule {
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
	if va[0] == "CompatibilityNote" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CompatibilityNote_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCompatibilityNote[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCompatibilityNote[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCompatibilityNote {
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
	if tok == "Domain" { errs += loadDomain(act,ln,pos,lno,flag,nm) }
	if tok == "Kernel" { errs += loadKernel(act,ln,pos,lno,flag,nm) }
	if tok == "KernelParam" { errs += loadKernelParam(act,ln,pos,lno,flag,nm) }
	if tok == "KernelOp" { errs += loadKernelOp(act,ln,pos,lno,flag,nm) }
	if tok == "Optimization" { errs += loadOptimization(act,ln,pos,lno,flag,nm) }
	if tok == "ScheduleOp" { errs += loadScheduleOp(act,ln,pos,lno,flag,nm) }
	if tok == "MemLayout" { errs += loadMemLayout(act,ln,pos,lno,flag,nm) }
	if tok == "Fusion" { errs += loadFusion(act,ln,pos,lno,flag,nm) }
	if tok == "Model" { errs += loadModel(act,ln,pos,lno,flag,nm) }
	if tok == "Block" { errs += loadBlock(act,ln,pos,lno,flag,nm) }
	if tok == "BlockParam" { errs += loadBlockParam(act,ln,pos,lno,flag,nm) }
	if tok == "BlockInstance" { errs += loadBlockInstance(act,ln,pos,lno,flag,nm) }
	if tok == "ParamValue" { errs += loadParamValue(act,ln,pos,lno,flag,nm) }
	if tok == "Tensor" { errs += loadTensor(act,ln,pos,lno,flag,nm) }
	if tok == "Layer" { errs += loadLayer(act,ln,pos,lno,flag,nm) }
	if tok == "Op" { errs += loadOp(act,ln,pos,lno,flag,nm) }
	if tok == "Arg" { errs += loadArg(act,ln,pos,lno,flag,nm) }
	if tok == "AttentionOp" { errs += loadAttentionOp(act,ln,pos,lno,flag,nm) }
	if tok == "AttentionFreeOp" { errs += loadAttentionFreeOp(act,ln,pos,lno,flag,nm) }
	if tok == "RecurrentFormulation" { errs += loadRecurrentFormulation(act,ln,pos,lno,flag,nm) }
	if tok == "GraphOp" { errs += loadGraphOp(act,ln,pos,lno,flag,nm) }
	if tok == "GraphLearningOp" { errs += loadGraphLearningOp(act,ln,pos,lno,flag,nm) }
	if tok == "StatefulOp" { errs += loadStatefulOp(act,ln,pos,lno,flag,nm) }
	if tok == "LTCOp" { errs += loadLTCOp(act,ln,pos,lno,flag,nm) }
	if tok == "ODEOp" { errs += loadODEOp(act,ln,pos,lno,flag,nm) }
	if tok == "SDEOp" { errs += loadSDEOp(act,ln,pos,lno,flag,nm) }
	if tok == "PDEOp" { errs += loadPDEOp(act,ln,pos,lno,flag,nm) }
	if tok == "MemoryOp" { errs += loadMemoryOp(act,ln,pos,lno,flag,nm) }
	if tok == "ExpertRoutingOp" { errs += loadExpertRoutingOp(act,ln,pos,lno,flag,nm) }
	if tok == "CapacityAwareRoutingOp" { errs += loadCapacityAwareRoutingOp(act,ln,pos,lno,flag,nm) }
	if tok == "SamplingOp" { errs += loadSamplingOp(act,ln,pos,lno,flag,nm) }
	if tok == "StochasticOp" { errs += loadStochasticOp(act,ln,pos,lno,flag,nm) }
	if tok == "DynamicRoutingOp" { errs += loadDynamicRoutingOp(act,ln,pos,lno,flag,nm) }
	if tok == "DynamicOp" { errs += loadDynamicOp(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralProgramOp" { errs += loadNeuralProgramOp(act,ln,pos,lno,flag,nm) }
	if tok == "WeightGenerationOp" { errs += loadWeightGenerationOp(act,ln,pos,lno,flag,nm) }
	if tok == "ArchitectureSearchOp" { errs += loadArchitectureSearchOp(act,ln,pos,lno,flag,nm) }
	if tok == "ContinuousDepthOp" { errs += loadContinuousDepthOp(act,ln,pos,lno,flag,nm) }
	if tok == "OpTypeRule" { errs += loadOpTypeRule(act,ln,pos,lno,flag,nm) }
	if tok == "ArgRoleRule" { errs += loadArgRoleRule(act,ln,pos,lno,flag,nm) }
	if tok == "Config" { errs += loadConfig(act,ln,pos,lno,flag,nm) }
	if tok == "Schedule" { errs += loadSchedule(act,ln,pos,lno,flag,nm) }
	if tok == "ControlFlow" { errs += loadControlFlow(act,ln,pos,lno,flag,nm) }
	if tok == "Condition" { errs += loadCondition(act,ln,pos,lno,flag,nm) }
	if tok == "Branch" { errs += loadBranch(act,ln,pos,lno,flag,nm) }
	if tok == "GraphTensor" { errs += loadGraphTensor(act,ln,pos,lno,flag,nm) }
	if tok == "StateTransfer" { errs += loadStateTransfer(act,ln,pos,lno,flag,nm) }
	if tok == "ContinuousLayer" { errs += loadContinuousLayer(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpace" { errs += loadSearchSpace(act,ln,pos,lno,flag,nm) }
	if tok == "SearchOp" { errs += loadSearchOp(act,ln,pos,lno,flag,nm) }
	if tok == "SearchEdge" { errs += loadSearchEdge(act,ln,pos,lno,flag,nm) }
	if tok == "ArchitectureParam" { errs += loadArchitectureParam(act,ln,pos,lno,flag,nm) }
	if tok == "ArchitectureGradient" { errs += loadArchitectureGradient(act,ln,pos,lno,flag,nm) }
	if tok == "HyperNetwork" { errs += loadHyperNetwork(act,ln,pos,lno,flag,nm) }
	if tok == "WeightGenerator" { errs += loadWeightGenerator(act,ln,pos,lno,flag,nm) }
	if tok == "MemoryMatrix" { errs += loadMemoryMatrix(act,ln,pos,lno,flag,nm) }
	if tok == "DifferentiableProgram" { errs += loadDifferentiableProgram(act,ln,pos,lno,flag,nm) }
	if tok == "ProgramInstruction" { errs += loadProgramInstruction(act,ln,pos,lno,flag,nm) }
	if tok == "MetaLearner" { errs += loadMetaLearner(act,ln,pos,lno,flag,nm) }
	if tok == "SparseExpertSystem" { errs += loadSparseExpertSystem(act,ln,pos,lno,flag,nm) }
	if tok == "CapacityAwareRouter" { errs += loadCapacityAwareRouter(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralSDE" { errs += loadNeuralSDE(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralPDE" { errs += loadNeuralPDE(act,ln,pos,lno,flag,nm) }
	if tok == "DynamicGraphNetwork" { errs += loadDynamicGraphNetwork(act,ln,pos,lno,flag,nm) }
	if tok == "GraphLearner" { errs += loadGraphLearner(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralProgram" { errs += loadNeuralProgram(act,ln,pos,lno,flag,nm) }
	if tok == "ProgInstruction" { errs += loadProgInstruction(act,ln,pos,lno,flag,nm) }
	if tok == "LiquidNetwork" { errs += loadLiquidNetwork(act,ln,pos,lno,flag,nm) }
	if tok == "SymbolicShape" { errs += loadSymbolicShape(act,ln,pos,lno,flag,nm) }
	if tok == "JITCompiler" { errs += loadJITCompiler(act,ln,pos,lno,flag,nm) }
	if tok == "TargetRule" { errs += loadTargetRule(act,ln,pos,lno,flag,nm) }
	if tok == "DtypeRule" { errs += loadDtypeRule(act,ln,pos,lno,flag,nm) }
	if tok == "FlagRule" { errs += loadFlagRule(act,ln,pos,lno,flag,nm) }
	if tok == "DistributionRule" { errs += loadDistributionRule(act,ln,pos,lno,flag,nm) }
	if tok == "SearchStrategyRule" { errs += loadSearchStrategyRule(act,ln,pos,lno,flag,nm) }
	if tok == "DynamicGraphRule" { errs += loadDynamicGraphRule(act,ln,pos,lno,flag,nm) }
	if tok == "CompatibilityNote" { errs += loadCompatibilityNote(act,ln,pos,lno,flag,nm) }
	return errs
}

