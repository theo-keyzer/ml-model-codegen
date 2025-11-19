package main

import (
	"strings"
	"fmt"
	"strconv"
)

type ActT struct {
	index       map[string]int
	ApDomain [] *KpDomain
	ApHardware [] *KpHardware
	ApGPU [] *KpGPU
	ApKernel [] *KpKernel
	ApDynamicKernel [] *KpDynamicKernel
	ApKernelParam [] *KpKernelParam
	ApKernelOp [] *KpKernelOp
	ApOptimization [] *KpOptimization
	ApFusion [] *KpFusion
	ApFramework [] *KpFramework
	ApPyTorch [] *KpPyTorch
	ApSearchSpace [] *KpSearchSpace
	ApSearchSpaceCode [] *KpSearchSpaceCode
	ApArchitectureParam [] *KpArchitectureParam
	ApSearchOp [] *KpSearchOp
	ApArchitectureGradient [] *KpArchitectureGradient
	ApExpertSystem [] *KpExpertSystem
	ApExpert [] *KpExpert
	ApRouterNetwork [] *KpRouterNetwork
	ApExpertRoutingOp [] *KpExpertRoutingOp
	ApCapacityAwareRoutingOp [] *KpCapacityAwareRoutingOp
	ApSparseExpertSystem [] *KpSparseExpertSystem
	ApContinuousModel [] *KpContinuousModel
	ApODEOp [] *KpODEOp
	ApNeuralODE [] *KpNeuralODE
	ApSDEOp [] *KpSDEOp
	ApNeuralSDE [] *KpNeuralSDE
	ApPDEOp [] *KpPDEOp
	ApNeuralPDE [] *KpNeuralPDE
	ApContinuousDepthOp [] *KpContinuousDepthOp
	ApMetaLearner [] *KpMetaLearner
	ApHypernetwork [] *KpHypernetwork
	ApWeightGenerationOp [] *KpWeightGenerationOp
	ApDifferentiableProgram [] *KpDifferentiableProgram
	ApControlFlow [] *KpControlFlow
	ApCondition [] *KpCondition
	ApBranch [] *KpBranch
	ApModel [] *KpModel
	ApTensor [] *KpTensor
	ApLayer [] *KpLayer
	ApOp [] *KpOp
	ApArg [] *KpArg
	ApArchitectureSearchOp [] *KpArchitectureSearchOp
	ApSearchSpaceOp [] *KpSearchSpaceOp
	ApConfig [] *KpConfig
	ApSchedule [] *KpSchedule
	ApValidation [] *KpValidation
	ApDynamicConstraint [] *KpDynamicConstraint
	ApSearchMethodRule [] *KpSearchMethodRule
	ApRoutingStrategyRule [] *KpRoutingStrategyRule
	ApSolverRule [] *KpSolverRule
	ApProject [] *KpProject
	ApTargetConfig [] *KpTargetConfig
	ApBuildRule [] *KpBuildRule
	ApCodegenRule [] *KpCodegenRule
	ApStructuredChoice [] *KpStructuredChoice
	ApNumericRange [] *KpNumericRange
	ApParameterMap [] *KpParameterMap
	ApCostModel [] *KpCostModel
	ApExecutionContext [] *KpExecutionContext
	ApMemoryBudget [] *KpMemoryBudget
	ApDataType [] *KpDataType
	ApShapeConstraint [] *KpShapeConstraint
	ApCompatibilityRule [] *KpCompatibilityRule
	ApValidatedReference [] *KpValidatedReference
	ApMathExpression [] *KpMathExpression
	ApEnhancedArchitectureParam [] *KpEnhancedArchitectureParam
	ApKernelExecutionContext [] *KpKernelExecutionContext
	ApProjectValidation [] *KpProjectValidation
	ApEnumerationCatalog [] *KpEnumerationCatalog
	ApCodeBlock [] *KpCodeBlock
	ApClassCode [] *KpClassCode
	ApFunctionCode [] *KpFunctionCode
	ApMethodCode [] *KpMethodCode
	ApOperationDef [] *KpOperationDef
	ApOperationParam [] *KpOperationParam
	ApSearchMethod [] *KpSearchMethod
	ApTrainingConfig [] *KpTrainingConfig
	ApArchitecture [] *KpArchitecture
	ApTemplatePlaceholder [] *KpTemplatePlaceholder
	ApCodeDependency [] *KpCodeDependency
	ApCodegenTemplate [] *KpCodegenTemplate
	ApCodeBlockReference [] *KpCodeBlockReference
	ApActorTemplate [] *KpActorTemplate
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
	for _, st := range act.ApKernel {

//  daml.unit:51, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Kernel.hardware:Hardware." + v,  "+", st.LineNo, "daml.unit:51, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOptimization {

//  daml.unit:98, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Optimization.target:Hardware." + v,  "+", st.LineNo, "daml.unit:98, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApFusion {

//  daml.unit:109, g_runh.act:180

		v, _ = st.Names["fused_kernel"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:Fusion.fused_kernel:Kernel." + v,  "*", st.LineNo, "daml.unit:109, g_runh.act:184" );
		st.Kfused_kernelp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:110, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Fusion.hardware:Hardware." + v,  "*", st.LineNo, "daml.unit:110, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSearchSpaceCode {

//  daml.unit:166, g_runh.act:180

		v, _ = st.Names["operations_dict"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:SearchSpaceCode.operations_dict:CodeBlock." + v,  "*", st.LineNo, "daml.unit:166, g_runh.act:184" );
		st.Koperations_dictp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:167, g_runh.act:180

		v, _ = st.Names["cell_code"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:SearchSpaceCode.cell_code:CodeBlock." + v,  "*", st.LineNo, "daml.unit:167, g_runh.act:184" );
		st.Kcell_codep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSearchOp {

//  daml.unit:190, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:SearchOp.hardware:Hardware." + v,  "*", st.LineNo, "daml.unit:190, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApExpertRoutingOp {

//  daml.unit:248, g_runh.act:180

		v, _ = st.Names["router"]
		err, res = fnd3(act, "RouterNetwork_" + v, v, "ref:ExpertRoutingOp.router:RouterNetwork." + v,  "+", st.LineNo, "daml.unit:248, g_runh.act:184" );
		st.Krouterp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:249, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:ExpertRoutingOp.hardware:Hardware." + v,  "*", st.LineNo, "daml.unit:249, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApWeightGenerationOp {

//  daml.unit:398, g_runh.act:180

		v, _ = st.Names["hypernetwork"]
		err, res = fnd3(act, "Hypernetwork_" + v, v, "ref:WeightGenerationOp.hypernetwork:Hypernetwork." + v,  "+", st.LineNo, "daml.unit:398, g_runh.act:184" );
		st.Khypernetworkp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApBranch {

//  daml.unit:442, g_runh.act:209

		v, _ = st.Names["condition"]
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Condition_" + v,v, "ref_link:Branch.condition:ControlFlow." + st.Parent + ".Condition." + v,  "*", st.LineNo, "daml.unit:442, g_runh.act:212" );
		st.Kconditionp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApModel {

//  daml.unit:457, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Model.hardware:Hardware." + v,  "*", st.LineNo, "daml.unit:457, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:458, g_runh.act:180

		v, _ = st.Names["framework"]
		err, res = fnd3(act, "Framework_" + v, v, "ref:Model.framework:Framework." + v,  "*", st.LineNo, "daml.unit:458, g_runh.act:184" );
		st.Kframeworkp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:459, g_runh.act:180

		v, _ = st.Names["search_space"]
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:Model.search_space:SearchSpace." + v,  "*", st.LineNo, "daml.unit:459, g_runh.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApLayer {

//  daml.unit:487, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:Layer.code_block:CodeBlock." + v,  "*", st.LineNo, "daml.unit:487, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOp {

//  daml.unit:499, g_runh.act:194

		v, _ = st.Names["op_type"]
		for id, child := range st.Childs {
			if child.TypeName() == v {
				st.Kop_typep = id
				break
			}
		}
//  daml.unit:502, g_runh.act:180

		v, _ = st.Names["kernel"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:Op.kernel:Kernel." + v,  "*", st.LineNo, "daml.unit:502, g_runh.act:184" );
		st.Kkernelp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:504, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:Op.code_block:CodeBlock." + v,  "*", st.LineNo, "daml.unit:504, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApArg {

//  daml.unit:512, g_runh.act:180

		v, _ = st.Names["tensor"]
		err, res = fnd3(act, "Tensor_" + v, v, "ref:Arg.tensor:Tensor." + v,  "+", st.LineNo, "daml.unit:512, g_runh.act:184" );
		st.Ktensorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApArchitectureSearchOp {

//  daml.unit:522, g_runh.act:180

		v, _ = st.Names["search_space"]
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:ArchitectureSearchOp.search_space:SearchSpace." + v,  "+", st.LineNo, "daml.unit:522, g_runh.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:523, g_runh.act:180

		v, _ = st.Names["search_method"]
		err, res = fnd3(act, "SearchOp_" + v, v, "ref:ArchitectureSearchOp.search_method:SearchOp." + v,  "+", st.LineNo, "daml.unit:523, g_runh.act:184" );
		st.Ksearch_methodp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSearchSpaceOp {

//  daml.unit:534, g_runh.act:180

		v, _ = st.Names["search_space"]
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:SearchSpaceOp.search_space:SearchSpace." + v,  "*", st.LineNo, "daml.unit:534, g_runh.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApConfig {

//  daml.unit:548, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Config.target:Hardware." + v,  "+", st.LineNo, "daml.unit:548, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSchedule {

//  daml.unit:559, g_runh.act:180

		v, _ = st.Names["layer"]
		err, res = fnd3(act, "Layer_" + v, v, "ref:Schedule.layer:Layer." + v,  "+", st.LineNo, "daml.unit:559, g_runh.act:184" );
		st.Klayerp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApValidation {

//  daml.unit:573, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Model_" + v, v, "ref:Validation.target:Model." + v,  "*", st.LineNo, "daml.unit:573, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApProject {

//  daml.unit:628, g_runh.act:180

		v, _ = st.Names["domain"]
		err, res = fnd3(act, "Domain_" + v, v, "ref:Project.domain:Domain." + v,  "*", st.LineNo, "daml.unit:628, g_runh.act:184" );
		st.Kdomainp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:629, g_runh.act:180

		v, _ = st.Names["model"]
		err, res = fnd3(act, "Model_" + v, v, "ref:Project.model:Model." + v,  "+", st.LineNo, "daml.unit:629, g_runh.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTargetConfig {

//  daml.unit:637, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:TargetConfig.hardware:Hardware." + v,  "+", st.LineNo, "daml.unit:637, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:639, g_runh.act:180

		v, _ = st.Names["config"]
		err, res = fnd3(act, "Config_" + v, v, "ref:TargetConfig.config:Config." + v,  "*", st.LineNo, "daml.unit:639, g_runh.act:184" );
		st.Kconfigp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApEnhancedArchitectureParam {

//  daml.unit:823, g_runh.act:180

		v, _ = st.Names["structured_choices"]
		err, res = fnd3(act, "StructuredChoice_" + v, v, "ref:EnhancedArchitectureParam.structured_choices:StructuredChoice." + v,  "*", st.LineNo, "daml.unit:823, g_runh.act:184" );
		st.Kstructured_choicesp = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:824, g_runh.act:180

		v, _ = st.Names["numeric_range"]
		err, res = fnd3(act, "NumericRange_" + v, v, "ref:EnhancedArchitectureParam.numeric_range:NumericRange." + v,  "*", st.LineNo, "daml.unit:824, g_runh.act:184" );
		st.Knumeric_rangep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApMethodCode {

//  daml.unit:928, g_runh.act:209

		v, _ = st.Names["parent_class"]
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_ClassCode_" + v,v, "ref_link:MethodCode.parent_class:CodeBlock." + st.Parent + ".ClassCode." + v,  "+", st.LineNo, "daml.unit:928, g_runh.act:212" );
		st.Kparent_classp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOperationDef {

//  daml.unit:946, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:OperationDef.code_block:CodeBlock." + v,  "+", st.LineNo, "daml.unit:946, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSearchMethod {

//  daml.unit:992, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:SearchMethod.code_block:CodeBlock." + v,  "*", st.LineNo, "daml.unit:992, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTrainingConfig {

//  daml.unit:1009, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:TrainingConfig.code_block:CodeBlock." + v,  "*", st.LineNo, "daml.unit:1009, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApArchitecture {

//  daml.unit:1035, g_runh.act:180

		v, _ = st.Names["main_code"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:Architecture.main_code:CodeBlock." + v,  "*", st.LineNo, "daml.unit:1035, g_runh.act:184" );
		st.Kmain_codep = res
		if (err == false) {
			errs += 1
		}
//  daml.unit:1036, g_runh.act:180

		v, _ = st.Names["config_code"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:Architecture.config_code:CodeBlock." + v,  "*", st.LineNo, "daml.unit:1036, g_runh.act:184" );
		st.Kconfig_codep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApCodeDependency {

//  daml.unit:1065, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:CodeDependency.code_block:CodeBlock." + v,  "+", st.LineNo, "daml.unit:1065, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApCodeBlockReference {

//  daml.unit:1095, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:CodeBlockReference.code_block:CodeBlock." + v,  "+", st.LineNo, "daml.unit:1095, g_runh.act:184" );
		st.Kcode_blockp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApActorTemplate {

//  daml.unit:1109, g_runh.act:180

		v, _ = st.Names["code_block"]
		err, res = fnd3(act, "CodeBlock_" + v, v, "ref:ActorTemplate.code_block:CodeBlock." + v,  "*", st.LineNo, "daml.unit:1109, g_runh.act:184" );
		st.Kcode_blockp = res
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
	for _, st := range act.ApDynamicKernel {

//  daml.unit:63, g_runh.act:262
	p = st.Me
	p = act.ApDynamicKernel[p].Kparentp
	if p >= 0 {
		st.Kcopy_kernelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:DynamicKernel.copy_kernel unresolved from key:DynamicKernel.dynamic_kernel:..x %s (+) > daml.unit:63, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  daml.unit:64, g_runh.act:248

	t := st.Kcopy_kernelp
	if t >= 0 {
		st.Khardwarep = act.ApKernel[t].Khardwarep
	} else if "+" != "*" {
		fmt.Printf("ref_copy:DynamicKernel.hardware unresolved from up_copy:DynamicKernel.copy_kernel:Kernel.x %s (+) > daml.unit:64, g_runh.act:254\n", st.LineNo)
		errs += 1
	}
//  daml.unit:65, g_runh.act:224

 
	if st.Khardwarep < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:DynamicKernel.gpu_target unresolved from ref_copy:DynamicKernel.hardware:Hardware %s > daml.unit:65, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApHardware[st.Khardwarep].MyName
		v, _ = st.Names["gpu_target"]
		err, res = fnd3(act, strconv.Itoa(st.Khardwarep) + "_GPU_" + v, v, "ref_child:DynamicKernel.gpu_target:Hardware." + parent + "." + v + " from ref_copy:DynamicKernel.hardware", "*", st.LineNo, "daml.unit:65, g_runh.act:236")
		st.Kgpu_targetp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApOp {

//  daml.unit:503, g_runh.act:224

 
	if st.Kkernelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Op.kernel_op unresolved from ref:Op.kernel:Kernel %s > daml.unit:503, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApKernel[st.Kkernelp].MyName
		v, _ = st.Names["kernel_op"]
		err, res = fnd3(act, strconv.Itoa(st.Kkernelp) + "_KernelOp_" + v, v, "ref_child:Op.kernel_op:Kernel." + parent + "." + v + " from ref:Op.kernel", "*", st.LineNo, "daml.unit:503, g_runh.act:236")
		st.Kkernel_opp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSearchSpaceOp {

//  daml.unit:535, g_runh.act:224

 
	if st.Ksearch_spacep < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:SearchSpaceOp.param_choices unresolved from ref:SearchSpaceOp.search_space:SearchSpace %s > daml.unit:535, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApSearchSpace[st.Ksearch_spacep].MyName
		v, _ = st.Names["param_choices"]
		err, res = fnd3(act, strconv.Itoa(st.Ksearch_spacep) + "_ArchitectureParam_" + v, v, "ref_child:SearchSpaceOp.param_choices:SearchSpace." + parent + "." + v + " from ref:SearchSpaceOp.search_space", "*", st.LineNo, "daml.unit:535, g_runh.act:236")
		st.Kparam_choicesp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSchedule {

//  daml.unit:560, g_runh.act:224

 
	if st.Klayerp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Schedule.op unresolved from ref:Schedule.layer:Layer %s > daml.unit:560, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApLayer[st.Klayerp].MyName
		v, _ = st.Names["op"]
		err, res = fnd3(act, strconv.Itoa(st.Klayerp) + "_Op_" + v, v, "ref_child:Schedule.op:Layer." + parent + "." + v + " from ref:Schedule.layer", "*", st.LineNo, "daml.unit:560, g_runh.act:236")
		st.Kopp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApBuildRule {

//  daml.unit:652, g_runh.act:262
	p = st.Me
	p = act.ApBuildRule[p].Kparentp
	if p >= 0 {
		st.Kprojectp = p
	} else if "*" != "*" {
		fmt.Printf("ref_copy:BuildRule.project unresolved from word:BuildRule.template:..x %s (*) > daml.unit:652, g_runh.act:275\n", st.LineNo)
		errs += 1
	}
//  daml.unit:653, g_runh.act:224

 
	if st.Kprojectp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:BuildRule.validate_against unresolved from up_copy:BuildRule.project:Project %s > daml.unit:653, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApProject[st.Kprojectp].MyName
		v, _ = st.Names["validate_against"]
		err, res = fnd3(act, strconv.Itoa(st.Kprojectp) + "_TargetConfig_" + v, v, "ref_child:BuildRule.validate_against:Project." + parent + "." + v + " from up_copy:BuildRule.project", "*", st.LineNo, "daml.unit:653, g_runh.act:236")
		st.Kvalidate_againstp = res
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
	if va[0] == "GPU" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["GPU_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApGPU[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApGPU[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApGPU {
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
	if va[0] == "DynamicKernel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DynamicKernel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDynamicKernel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDynamicKernel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDynamicKernel {
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
	if va[0] == "Framework" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Framework_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFramework[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFramework[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFramework {
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
	if va[0] == "PyTorch" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PyTorch_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPyTorch[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPyTorch[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPyTorch {
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
	if va[0] == "SearchSpaceCode" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchSpaceCode_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchSpaceCode[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchSpaceCode[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchSpaceCode {
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
	if va[0] == "ExpertSystem" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ExpertSystem_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApExpertSystem[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApExpertSystem[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApExpertSystem {
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
	if va[0] == "Expert" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Expert_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApExpert[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApExpert[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApExpert {
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
	if va[0] == "RouterNetwork" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["RouterNetwork_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApRouterNetwork[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApRouterNetwork[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApRouterNetwork {
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
	if va[0] == "ContinuousModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ContinuousModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApContinuousModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApContinuousModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApContinuousModel {
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
	if va[0] == "NeuralODE" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NeuralODE_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNeuralODE[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNeuralODE[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNeuralODE {
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
	if va[0] == "Hypernetwork" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Hypernetwork_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHypernetwork[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHypernetwork[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHypernetwork {
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
	if va[0] == "SearchSpaceOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchSpaceOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchSpaceOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchSpaceOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchSpaceOp {
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
	if va[0] == "DynamicConstraint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DynamicConstraint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDynamicConstraint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDynamicConstraint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDynamicConstraint {
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
	if va[0] == "SearchMethodRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchMethodRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchMethodRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchMethodRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchMethodRule {
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
	if va[0] == "RoutingStrategyRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["RoutingStrategyRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApRoutingStrategyRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApRoutingStrategyRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApRoutingStrategyRule {
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
	if va[0] == "SolverRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SolverRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSolverRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSolverRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSolverRule {
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
	if va[0] == "TargetConfig" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TargetConfig_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTargetConfig[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTargetConfig[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTargetConfig {
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
	if va[0] == "BuildRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["BuildRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBuildRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBuildRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBuildRule {
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
	if va[0] == "CodegenRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CodegenRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCodegenRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCodegenRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCodegenRule {
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
	if va[0] == "StructuredChoice" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["StructuredChoice_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApStructuredChoice[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApStructuredChoice[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApStructuredChoice {
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
	if va[0] == "NumericRange" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NumericRange_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNumericRange[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNumericRange[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNumericRange {
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
	if va[0] == "ParameterMap" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ParameterMap_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApParameterMap[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApParameterMap[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApParameterMap {
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
	if va[0] == "CostModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CostModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCostModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCostModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCostModel {
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
	if va[0] == "ExecutionContext" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ExecutionContext_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApExecutionContext[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApExecutionContext[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApExecutionContext {
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
	if va[0] == "MemoryBudget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MemoryBudget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMemoryBudget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMemoryBudget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMemoryBudget {
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
	if va[0] == "DataType" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DataType_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDataType[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDataType[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDataType {
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
	if va[0] == "ShapeConstraint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ShapeConstraint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApShapeConstraint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApShapeConstraint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApShapeConstraint {
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
	if va[0] == "CompatibilityRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CompatibilityRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCompatibilityRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCompatibilityRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCompatibilityRule {
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
	if va[0] == "ValidatedReference" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ValidatedReference_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApValidatedReference[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApValidatedReference[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApValidatedReference {
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
	if va[0] == "MathExpression" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MathExpression_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMathExpression[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMathExpression[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMathExpression {
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
	if va[0] == "EnhancedArchitectureParam" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnhancedArchitectureParam_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnhancedArchitectureParam[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnhancedArchitectureParam[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnhancedArchitectureParam {
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
	if va[0] == "KernelExecutionContext" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["KernelExecutionContext_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApKernelExecutionContext[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApKernelExecutionContext[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApKernelExecutionContext {
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
	if va[0] == "ProjectValidation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ProjectValidation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApProjectValidation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApProjectValidation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApProjectValidation {
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
	if va[0] == "EnumerationCatalog" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnumerationCatalog_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnumerationCatalog[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnumerationCatalog[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnumerationCatalog {
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
	if va[0] == "CodeBlock" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CodeBlock_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCodeBlock[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCodeBlock[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCodeBlock {
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
	if va[0] == "ClassCode" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ClassCode_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApClassCode[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApClassCode[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApClassCode {
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
	if va[0] == "FunctionCode" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FunctionCode_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFunctionCode[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFunctionCode[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFunctionCode {
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
	if va[0] == "MethodCode" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MethodCode_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMethodCode[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMethodCode[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMethodCode {
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
	if va[0] == "OperationDef" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OperationDef_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOperationDef[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOperationDef[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOperationDef {
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
	if va[0] == "OperationParam" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OperationParam_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOperationParam[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOperationParam[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOperationParam {
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
	if va[0] == "SearchMethod" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SearchMethod_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSearchMethod[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSearchMethod[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSearchMethod {
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
	if va[0] == "TrainingConfig" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TrainingConfig_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTrainingConfig[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTrainingConfig[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTrainingConfig {
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
	if va[0] == "Architecture" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Architecture_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApArchitecture[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApArchitecture[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApArchitecture {
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
	if va[0] == "TemplatePlaceholder" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TemplatePlaceholder_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTemplatePlaceholder[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTemplatePlaceholder[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTemplatePlaceholder {
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
	if va[0] == "CodeDependency" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CodeDependency_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCodeDependency[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCodeDependency[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCodeDependency {
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
	if va[0] == "CodegenTemplate" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CodegenTemplate_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCodegenTemplate[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCodegenTemplate[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCodegenTemplate {
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
	if va[0] == "CodeBlockReference" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["CodeBlockReference_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApCodeBlockReference[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApCodeBlockReference[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApCodeBlockReference {
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
	if va[0] == "ActorTemplate" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ActorTemplate_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApActorTemplate[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApActorTemplate[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApActorTemplate {
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
	if tok == "Hardware" { errs += loadHardware(act,ln,pos,lno,flag,nm) }
	if tok == "GPU" { errs += loadGPU(act,ln,pos,lno,flag,nm) }
	if tok == "Kernel" { errs += loadKernel(act,ln,pos,lno,flag,nm) }
	if tok == "DynamicKernel" { errs += loadDynamicKernel(act,ln,pos,lno,flag,nm) }
	if tok == "KernelParam" { errs += loadKernelParam(act,ln,pos,lno,flag,nm) }
	if tok == "KernelOp" { errs += loadKernelOp(act,ln,pos,lno,flag,nm) }
	if tok == "Optimization" { errs += loadOptimization(act,ln,pos,lno,flag,nm) }
	if tok == "Fusion" { errs += loadFusion(act,ln,pos,lno,flag,nm) }
	if tok == "Framework" { errs += loadFramework(act,ln,pos,lno,flag,nm) }
	if tok == "PyTorch" { errs += loadPyTorch(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpace" { errs += loadSearchSpace(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpaceCode" { errs += loadSearchSpaceCode(act,ln,pos,lno,flag,nm) }
	if tok == "ArchitectureParam" { errs += loadArchitectureParam(act,ln,pos,lno,flag,nm) }
	if tok == "SearchOp" { errs += loadSearchOp(act,ln,pos,lno,flag,nm) }
	if tok == "ArchitectureGradient" { errs += loadArchitectureGradient(act,ln,pos,lno,flag,nm) }
	if tok == "ExpertSystem" { errs += loadExpertSystem(act,ln,pos,lno,flag,nm) }
	if tok == "Expert" { errs += loadExpert(act,ln,pos,lno,flag,nm) }
	if tok == "RouterNetwork" { errs += loadRouterNetwork(act,ln,pos,lno,flag,nm) }
	if tok == "ExpertRoutingOp" { errs += loadExpertRoutingOp(act,ln,pos,lno,flag,nm) }
	if tok == "CapacityAwareRoutingOp" { errs += loadCapacityAwareRoutingOp(act,ln,pos,lno,flag,nm) }
	if tok == "SparseExpertSystem" { errs += loadSparseExpertSystem(act,ln,pos,lno,flag,nm) }
	if tok == "ContinuousModel" { errs += loadContinuousModel(act,ln,pos,lno,flag,nm) }
	if tok == "ODEOp" { errs += loadODEOp(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralODE" { errs += loadNeuralODE(act,ln,pos,lno,flag,nm) }
	if tok == "SDEOp" { errs += loadSDEOp(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralSDE" { errs += loadNeuralSDE(act,ln,pos,lno,flag,nm) }
	if tok == "PDEOp" { errs += loadPDEOp(act,ln,pos,lno,flag,nm) }
	if tok == "NeuralPDE" { errs += loadNeuralPDE(act,ln,pos,lno,flag,nm) }
	if tok == "ContinuousDepthOp" { errs += loadContinuousDepthOp(act,ln,pos,lno,flag,nm) }
	if tok == "MetaLearner" { errs += loadMetaLearner(act,ln,pos,lno,flag,nm) }
	if tok == "Hypernetwork" { errs += loadHypernetwork(act,ln,pos,lno,flag,nm) }
	if tok == "WeightGenerationOp" { errs += loadWeightGenerationOp(act,ln,pos,lno,flag,nm) }
	if tok == "DifferentiableProgram" { errs += loadDifferentiableProgram(act,ln,pos,lno,flag,nm) }
	if tok == "ControlFlow" { errs += loadControlFlow(act,ln,pos,lno,flag,nm) }
	if tok == "Condition" { errs += loadCondition(act,ln,pos,lno,flag,nm) }
	if tok == "Branch" { errs += loadBranch(act,ln,pos,lno,flag,nm) }
	if tok == "Model" { errs += loadModel(act,ln,pos,lno,flag,nm) }
	if tok == "Tensor" { errs += loadTensor(act,ln,pos,lno,flag,nm) }
	if tok == "Layer" { errs += loadLayer(act,ln,pos,lno,flag,nm) }
	if tok == "Op" { errs += loadOp(act,ln,pos,lno,flag,nm) }
	if tok == "Arg" { errs += loadArg(act,ln,pos,lno,flag,nm) }
	if tok == "ArchitectureSearchOp" { errs += loadArchitectureSearchOp(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpaceOp" { errs += loadSearchSpaceOp(act,ln,pos,lno,flag,nm) }
	if tok == "Config" { errs += loadConfig(act,ln,pos,lno,flag,nm) }
	if tok == "Schedule" { errs += loadSchedule(act,ln,pos,lno,flag,nm) }
	if tok == "Validation" { errs += loadValidation(act,ln,pos,lno,flag,nm) }
	if tok == "DynamicConstraint" { errs += loadDynamicConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "SearchMethodRule" { errs += loadSearchMethodRule(act,ln,pos,lno,flag,nm) }
	if tok == "RoutingStrategyRule" { errs += loadRoutingStrategyRule(act,ln,pos,lno,flag,nm) }
	if tok == "SolverRule" { errs += loadSolverRule(act,ln,pos,lno,flag,nm) }
	if tok == "Project" { errs += loadProject(act,ln,pos,lno,flag,nm) }
	if tok == "TargetConfig" { errs += loadTargetConfig(act,ln,pos,lno,flag,nm) }
	if tok == "BuildRule" { errs += loadBuildRule(act,ln,pos,lno,flag,nm) }
	if tok == "CodegenRule" { errs += loadCodegenRule(act,ln,pos,lno,flag,nm) }
	if tok == "StructuredChoice" { errs += loadStructuredChoice(act,ln,pos,lno,flag,nm) }
	if tok == "NumericRange" { errs += loadNumericRange(act,ln,pos,lno,flag,nm) }
	if tok == "ParameterMap" { errs += loadParameterMap(act,ln,pos,lno,flag,nm) }
	if tok == "CostModel" { errs += loadCostModel(act,ln,pos,lno,flag,nm) }
	if tok == "ExecutionContext" { errs += loadExecutionContext(act,ln,pos,lno,flag,nm) }
	if tok == "MemoryBudget" { errs += loadMemoryBudget(act,ln,pos,lno,flag,nm) }
	if tok == "DataType" { errs += loadDataType(act,ln,pos,lno,flag,nm) }
	if tok == "ShapeConstraint" { errs += loadShapeConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "CompatibilityRule" { errs += loadCompatibilityRule(act,ln,pos,lno,flag,nm) }
	if tok == "ValidatedReference" { errs += loadValidatedReference(act,ln,pos,lno,flag,nm) }
	if tok == "MathExpression" { errs += loadMathExpression(act,ln,pos,lno,flag,nm) }
	if tok == "EnhancedArchitectureParam" { errs += loadEnhancedArchitectureParam(act,ln,pos,lno,flag,nm) }
	if tok == "KernelExecutionContext" { errs += loadKernelExecutionContext(act,ln,pos,lno,flag,nm) }
	if tok == "ProjectValidation" { errs += loadProjectValidation(act,ln,pos,lno,flag,nm) }
	if tok == "EnumerationCatalog" { errs += loadEnumerationCatalog(act,ln,pos,lno,flag,nm) }
	if tok == "CodeBlock" { errs += loadCodeBlock(act,ln,pos,lno,flag,nm) }
	if tok == "ClassCode" { errs += loadClassCode(act,ln,pos,lno,flag,nm) }
	if tok == "FunctionCode" { errs += loadFunctionCode(act,ln,pos,lno,flag,nm) }
	if tok == "MethodCode" { errs += loadMethodCode(act,ln,pos,lno,flag,nm) }
	if tok == "OperationDef" { errs += loadOperationDef(act,ln,pos,lno,flag,nm) }
	if tok == "OperationParam" { errs += loadOperationParam(act,ln,pos,lno,flag,nm) }
	if tok == "SearchMethod" { errs += loadSearchMethod(act,ln,pos,lno,flag,nm) }
	if tok == "TrainingConfig" { errs += loadTrainingConfig(act,ln,pos,lno,flag,nm) }
	if tok == "Architecture" { errs += loadArchitecture(act,ln,pos,lno,flag,nm) }
	if tok == "TemplatePlaceholder" { errs += loadTemplatePlaceholder(act,ln,pos,lno,flag,nm) }
	if tok == "CodeDependency" { errs += loadCodeDependency(act,ln,pos,lno,flag,nm) }
	if tok == "CodegenTemplate" { errs += loadCodegenTemplate(act,ln,pos,lno,flag,nm) }
	if tok == "CodeBlockReference" { errs += loadCodeBlockReference(act,ln,pos,lno,flag,nm) }
	if tok == "ActorTemplate" { errs += loadActorTemplate(act,ln,pos,lno,flag,nm) }
	return errs
}

