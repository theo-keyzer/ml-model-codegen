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
	ApTSU [] *KpTSU
	ApKernel [] *KpKernel
	ApTSUKernel [] *KpTSUKernel
	ApKernelParam [] *KpKernelParam
	ApKernelOp [] *KpKernelOp
	ApOptimization [] *KpOptimization
	ApFusion [] *KpFusion
	ApFramework [] *KpFramework
	ApTHRML [] *KpTHRML
	ApPGMSchema [] *KpPGMSchema
	ApModel [] *KpModel
	ApTensor [] *KpTensor
	ApLayer [] *KpLayer
	ApOp [] *KpOp
	ApArg [] *KpArg
	ApSamplingOp [] *KpSamplingOp
	ApTSUSamplingOp [] *KpTSUSamplingOp
	ApBlockGibbsOp [] *KpBlockGibbsOp
	ApEnergyFunction [] *KpEnergyFunction
	ApEnergyFactor [] *KpEnergyFactor
	ApTSUCompilation [] *KpTSUCompilation
	ApTSUSubstrateModel [] *KpTSUSubstrateModel
	ApConfig [] *KpConfig
	ApSchedule [] *KpSchedule
	ApThermodynamicSimulation [] *KpThermodynamicSimulation
	ApValidation [] *KpValidation
	ApPhysicalConstraint [] *KpPhysicalConstraint
	ApDistributionRule [] *KpDistributionRule
	ApSamplingAlgorithmRule [] *KpSamplingAlgorithmRule
	ApBackendRule [] *KpBackendRule
	ApHardwareRule [] *KpHardwareRule
	ApProject [] *KpProject
	ApTargetConfig [] *KpTargetConfig
	ApBuildRule [] *KpBuildRule
	ApSearchSpace [] *KpSearchSpace
	ApSearchParameter [] *KpSearchParameter
	ApScheduleParameter [] *KpScheduleParameter
	ApEvolutionStrategy [] *KpEvolutionStrategy
	ApFitnessFunction [] *KpFitnessFunction
	ApAdaptiveParameter [] *KpAdaptiveParameter
	ApPerformanceMetric [] *KpPerformanceMetric
	ApExperimentTrial [] *KpExperimentTrial
	ApTrialMetrics [] *KpTrialMetrics
	ApOptimizationRun [] *KpOptimizationRun
	ApCheckpoint [] *KpCheckpoint
	ApHyperParameterRule [] *KpHyperParameterRule
	ApMetaLearning [] *KpMetaLearning
	ApAutoTuneConfig [] *KpAutoTuneConfig
	ApSensitivityAnalysis [] *KpSensitivityAnalysis
	ApEnsembleStrategy [] *KpEnsembleStrategy
	ApOptimizationPattern [] *KpOptimizationPattern
	ApTransferLearning [] *KpTransferLearning
	ApMultiObjective [] *KpMultiObjective
	ApParetoFront [] *KpParetoFront
	ApDiagnosticRule [] *KpDiagnosticRule
	ApFailureMode [] *KpFailureMode
	ApResourceBudget [] *KpResourceBudget
	ApAdaptiveScheduling [] *KpAdaptiveScheduling
	ApMultiStateModel [] *KpMultiStateModel
	ApLoopyPGM [] *KpLoopyPGM
	ApVariationalModel [] *KpVariationalModel
	ApHybridSampling [] *KpHybridSampling
	ApPottsOp [] *KpPottsOp
	ApPottsKernel [] *KpPottsKernel
	ApMultiStateOp [] *KpMultiStateOp
	ApClusterSamplingOp [] *KpClusterSamplingOp
	ApVariationalOp [] *KpVariationalOp
	ApBeliefPropagationOp [] *KpBeliefPropagationOp
	ApMultiStateEnergyFactor [] *KpMultiStateEnergyFactor
	ApVariationalFactor [] *KpVariationalFactor
	ApDegenerateEnergy [] *KpDegenerateEnergy
	ApMultiLevelTSU [] *KpMultiLevelTSU
	ApAdvancedTSUCompilation [] *KpAdvancedTSUCompilation
	ApNoiseModel [] *KpNoiseModel
	ApHybridConfig [] *KpHybridConfig
	ApELBOTracking [] *KpELBOTracking
	ApErgodicityValidation [] *KpErgodicityValidation
	ApMultiStateConstraint [] *KpMultiStateConstraint
	ApDegeneracySimulation [] *KpDegeneracySimulation
	ApClusterKernel [] *KpClusterKernel
	ApVariationalKernel [] *KpVariationalKernel
	ApMultiStateDistributionRule [] *KpMultiStateDistributionRule
	ApAdvancedSamplingAlgorithmRule [] *KpAdvancedSamplingAlgorithmRule
	ApVIAlgorithmRule [] *KpVIAlgorithmRule
	ApMultiStateProject [] *KpMultiStateProject
	ApAutoTuneExtension [] *KpAutoTuneExtension
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

//  tsu.unit:51, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Kernel.hardware:Hardware." + v,  "+", st.LineNo, "tsu.unit:51, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOptimization {

//  tsu.unit:101, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Optimization.target:Hardware." + v,  "+", st.LineNo, "tsu.unit:101, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApFusion {

//  tsu.unit:112, g_runh.act:180

		v, _ = st.Names["fused_kernel"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:Fusion.fused_kernel:Kernel." + v,  "*", st.LineNo, "tsu.unit:112, g_runh.act:184" );
		st.Kfused_kernelp = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:113, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Fusion.hardware:Hardware." + v,  "*", st.LineNo, "tsu.unit:113, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApModel {

//  tsu.unit:167, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Model.hardware:Hardware." + v,  "*", st.LineNo, "tsu.unit:167, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:168, g_runh.act:180

		v, _ = st.Names["framework"]
		err, res = fnd3(act, "Framework_" + v, v, "ref:Model.framework:Framework." + v,  "*", st.LineNo, "tsu.unit:168, g_runh.act:184" );
		st.Kframeworkp = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:169, g_runh.act:180

		v, _ = st.Names["pgm_schema"]
		err, res = fnd3(act, "PGMSchema_" + v, v, "ref:Model.pgm_schema:PGMSchema." + v,  "*", st.LineNo, "tsu.unit:169, g_runh.act:184" );
		st.Kpgm_schemap = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTensor {

//  tsu.unit:183, g_runh.act:180

		v, _ = st.Names["distribution"]
		err, res = fnd3(act, "DistributionRule_" + v, v, "ref:Tensor.distribution:DistributionRule." + v,  "*", st.LineNo, "tsu.unit:183, g_runh.act:184" );
		st.Kdistributionp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOp {

//  tsu.unit:203, g_runh.act:194

		v, _ = st.Names["op_type"]
		for id, child := range st.Childs {
			if child.TypeName() == v {
				st.Kop_typep = id
				break
			}
		}
//  tsu.unit:206, g_runh.act:180

		v, _ = st.Names["kernel"]
		err, res = fnd3(act, "Kernel_" + v, v, "ref:Op.kernel:Kernel." + v,  "*", st.LineNo, "tsu.unit:206, g_runh.act:184" );
		st.Kkernelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApArg {

//  tsu.unit:215, g_runh.act:180

		v, _ = st.Names["tensor"]
		err, res = fnd3(act, "Tensor_" + v, v, "ref:Arg.tensor:Tensor." + v,  "+", st.LineNo, "tsu.unit:215, g_runh.act:184" );
		st.Ktensorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSamplingOp {

//  tsu.unit:231, g_runh.act:180

		v, _ = st.Names["energy_fn_ref"]
		err, res = fnd3(act, "EnergyFunction_" + v, v, "ref:SamplingOp.energy_fn_ref:EnergyFunction." + v,  "*", st.LineNo, "tsu.unit:231, g_runh.act:184" );
		st.Kenergy_fn_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApEnergyFunction {

//  tsu.unit:263, g_runh.act:180

		v, _ = st.Names["params"]
		err, res = fnd3(act, "Tensor_" + v, v, "ref:EnergyFunction.params:Tensor." + v,  "*", st.LineNo, "tsu.unit:263, g_runh.act:184" );
		st.Kparamsp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApEnergyFactor {

//  tsu.unit:273, g_runh.act:180

		v, _ = st.Names["variables"]
		err, res = fnd3(act, "Tensor_" + v, v, "ref:EnergyFactor.variables:Tensor." + v,  "+", st.LineNo, "tsu.unit:273, g_runh.act:184" );
		st.Kvariablesp = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:276, g_runh.act:180

		v, _ = st.Names["param_tensor"]
		err, res = fnd3(act, "Tensor_" + v, v, "ref:EnergyFactor.param_tensor:Tensor." + v,  "*", st.LineNo, "tsu.unit:276, g_runh.act:184" );
		st.Kparam_tensorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTSUCompilation {

//  tsu.unit:290, g_runh.act:180

		v, _ = st.Names["source"]
		err, res = fnd3(act, "Model_" + v, v, "ref:TSUCompilation.source:Model." + v,  "+", st.LineNo, "tsu.unit:290, g_runh.act:184" );
		st.Ksourcep = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:291, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:TSUCompilation.hardware:Hardware." + v,  "+", st.LineNo, "tsu.unit:291, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApConfig {

//  tsu.unit:322, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Config.target:Hardware." + v,  "+", st.LineNo, "tsu.unit:322, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSchedule {

//  tsu.unit:332, g_runh.act:180

		v, _ = st.Names["layer"]
		err, res = fnd3(act, "Layer_" + v, v, "ref:Schedule.layer:Layer." + v,  "+", st.LineNo, "tsu.unit:332, g_runh.act:184" );
		st.Klayerp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApValidation {

//  tsu.unit:356, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Model_" + v, v, "ref:Validation.target:Model." + v,  "*", st.LineNo, "tsu.unit:356, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApProject {

//  tsu.unit:422, g_runh.act:180

		v, _ = st.Names["domain"]
		err, res = fnd3(act, "Domain_" + v, v, "ref:Project.domain:Domain." + v,  "*", st.LineNo, "tsu.unit:422, g_runh.act:184" );
		st.Kdomainp = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:423, g_runh.act:180

		v, _ = st.Names["model"]
		err, res = fnd3(act, "Model_" + v, v, "ref:Project.model:Model." + v,  "+", st.LineNo, "tsu.unit:423, g_runh.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTargetConfig {

//  tsu.unit:431, g_runh.act:180

		v, _ = st.Names["hardware"]
		err, res = fnd3(act, "Hardware_" + v, v, "ref:TargetConfig.hardware:Hardware." + v,  "+", st.LineNo, "tsu.unit:431, g_runh.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
//  tsu.unit:433, g_runh.act:180

		v, _ = st.Names["config"]
		err, res = fnd3(act, "Config_" + v, v, "ref:TargetConfig.config:Config." + v,  "*", st.LineNo, "tsu.unit:433, g_runh.act:184" );
		st.Kconfigp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSearchSpace {

//  tsu-auto.unit:10, g_runh.act:180

		v, _ = st.Names["target"]
		err, res = fnd3(act, "Config_" + v, v, "ref:SearchSpace.target:Config." + v,  "*", st.LineNo, "tsu-auto.unit:10, g_runh.act:184" );
		st.Ktargetp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApEvolutionStrategy {

//  tsu-auto.unit:47, g_runh.act:180

		v, _ = st.Names["search_space"]
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:EvolutionStrategy.search_space:SearchSpace." + v,  "+", st.LineNo, "tsu-auto.unit:47, g_runh.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApExperimentTrial {

//  tsu-auto.unit:96, g_runh.act:180

		v, _ = st.Names["strategy"]
		err, res = fnd3(act, "EvolutionStrategy_" + v, v, "ref:ExperimentTrial.strategy:EvolutionStrategy." + v,  "+", st.LineNo, "tsu-auto.unit:96, g_runh.act:184" );
		st.Kstrategyp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTrialMetrics {

//  tsu-auto.unit:108, g_runh.act:180

		v, _ = st.Names["metric"]
		err, res = fnd3(act, "PerformanceMetric_" + v, v, "ref:TrialMetrics.metric:PerformanceMetric." + v,  "+", st.LineNo, "tsu-auto.unit:108, g_runh.act:184" );
		st.Kmetricp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOptimizationRun {

//  tsu-auto.unit:118, g_runh.act:180

		v, _ = st.Names["strategy"]
		err, res = fnd3(act, "EvolutionStrategy_" + v, v, "ref:OptimizationRun.strategy:EvolutionStrategy." + v,  "+", st.LineNo, "tsu-auto.unit:118, g_runh.act:184" );
		st.Kstrategyp = res
		if (err == false) {
			errs += 1
		}
//  tsu-auto.unit:122, g_runh.act:180

		v, _ = st.Names["best_trial"]
		err, res = fnd3(act, "ExperimentTrial_" + v, v, "ref:OptimizationRun.best_trial:ExperimentTrial." + v,  "*", st.LineNo, "tsu-auto.unit:122, g_runh.act:184" );
		st.Kbest_trialp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApHyperParameterRule {

//  tsu-auto.unit:145, g_runh.act:180

		v, _ = st.Names["learned_from"]
		err, res = fnd3(act, "OptimizationRun_" + v, v, "ref:HyperParameterRule.learned_from:OptimizationRun." + v,  "*", st.LineNo, "tsu-auto.unit:145, g_runh.act:184" );
		st.Klearned_fromp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApAutoTuneConfig {

//  tsu-auto.unit:165, g_runh.act:180

		v, _ = st.Names["strategy"]
		err, res = fnd3(act, "EvolutionStrategy_" + v, v, "ref:AutoTuneConfig.strategy:EvolutionStrategy." + v,  "+", st.LineNo, "tsu-auto.unit:165, g_runh.act:184" );
		st.Kstrategyp = res
		if (err == false) {
			errs += 1
		}
//  tsu-auto.unit:168, g_runh.act:180

		v, _ = st.Names["opt_from"]
		err, res = fnd3(act, "OptimizationRun_" + v, v, "ref:AutoTuneConfig.opt_from:OptimizationRun." + v,  "*", st.LineNo, "tsu-auto.unit:168, g_runh.act:184" );
		st.Kopt_fromp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSensitivityAnalysis {

//  tsu-auto.unit:177, g_runh.act:180

		v, _ = st.Names["search_space"]
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:SensitivityAnalysis.search_space:SearchSpace." + v,  "+", st.LineNo, "tsu-auto.unit:177, g_runh.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOptimizationPattern {

//  tsu-auto.unit:207, g_runh.act:180

		v, _ = st.Names["discovered_by"]
		err, res = fnd3(act, "OptimizationRun_" + v, v, "ref:OptimizationPattern.discovered_by:OptimizationRun." + v,  "*", st.LineNo, "tsu-auto.unit:207, g_runh.act:184" );
		st.Kdiscovered_byp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApVariationalModel {

//  tsu-ext.unit:38, g_runh.act:180

		v, _ = st.Names["optimizer_ref"]
		err, res = fnd3(act, "EvolutionStrategy_" + v, v, "ref:VariationalModel.optimizer_ref:EvolutionStrategy." + v,  "*", st.LineNo, "tsu-ext.unit:38, g_runh.act:184" );
		st.Koptimizer_refp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApHybridSampling {

//  tsu-ext.unit:48, g_runh.act:180

		v, _ = st.Names["fusion_pattern"]
		err, res = fnd3(act, "Fusion_" + v, v, "ref:HybridSampling.fusion_pattern:Fusion." + v,  "*", st.LineNo, "tsu-ext.unit:48, g_runh.act:184" );
		st.Kfusion_patternp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApPottsOp {

//  tsu-ext.unit:62, g_runh.act:180

		v, _ = st.Names["energy_fn"]
		err, res = fnd3(act, "EnergyFunction_" + v, v, "ref:PottsOp.energy_fn:EnergyFunction." + v,  "+", st.LineNo, "tsu-ext.unit:62, g_runh.act:184" );
		st.Kenergy_fnp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApAutoTuneExtension {

//  tsu-ext.unit:310, g_runh.act:180

		v, _ = st.Names["multi_objective"]
		err, res = fnd3(act, "MultiObjective_" + v, v, "ref:AutoTuneExtension.multi_objective:MultiObjective." + v,  "*", st.LineNo, "tsu-ext.unit:310, g_runh.act:184" );
		st.Kmulti_objectivep = res
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
	for _, st := range act.ApTSUKernel {

//  tsu.unit:64, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kcopy_kernelp = p
	} else if "+" != "*" {
		fmt.Printf("ref_copy:TSUKernel.copy_kernel unresolved from key:TSUKernel.tsu_kernel:..x %s (+) > tsu.unit:64, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  tsu.unit:65, g_runh.act:248

	t := st.Kcopy_kernelp
	if t >= 0 {
		st.Khardwarep = act.ApKernel[t].Khardwarep
	} else if "+" != "*" {
		fmt.Printf("ref_copy:TSUKernel.hardware unresolved from up_copy:TSUKernel.copy_kernel:Kernel.x %s (+) > tsu.unit:65, g_runh.act:254\n", st.LineNo)
		errs += 1
	}
//  tsu.unit:66, g_runh.act:224

 
	if st.Khardwarep < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:TSUKernel.tsu_target unresolved from ref_copy:TSUKernel.hardware:Hardware %s > tsu.unit:66, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApHardware[st.Khardwarep].MyName
		v, _ = st.Names["tsu_target"]
		err, res = fnd3(act, strconv.Itoa(st.Khardwarep) + "_TSU_" + v, v, "ref_child:TSUKernel.tsu_target:Hardware." + parent + "." + v + " from ref_copy:TSUKernel.hardware", "+", st.LineNo, "tsu.unit:66, g_runh.act:236")
		st.Ktsu_targetp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApOp {

//  tsu.unit:207, g_runh.act:224

 
	if st.Kkernelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Op.kernel_op unresolved from ref:Op.kernel:Kernel %s > tsu.unit:207, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApKernel[st.Kkernelp].MyName
		v, _ = st.Names["kernel_op"]
		err, res = fnd3(act, strconv.Itoa(st.Kkernelp) + "_KernelOp_" + v, v, "ref_child:Op.kernel_op:Kernel." + parent + "." + v + " from ref:Op.kernel", "*", st.LineNo, "tsu.unit:207, g_runh.act:236")
		st.Kkernel_opp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApTSUCompilation {

//  tsu.unit:292, g_runh.act:224

 
	if st.Khardwarep < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:TSUCompilation.target unresolved from ref:TSUCompilation.hardware:Hardware %s > tsu.unit:292, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApHardware[st.Khardwarep].MyName
		v, _ = st.Names["target"]
		err, res = fnd3(act, strconv.Itoa(st.Khardwarep) + "_TSU_" + v, v, "ref_child:TSUCompilation.target:Hardware." + parent + "." + v + " from ref:TSUCompilation.hardware", "+", st.LineNo, "tsu.unit:292, g_runh.act:236")
		st.Ktargetp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApSchedule {

//  tsu.unit:333, g_runh.act:224

 
	if st.Klayerp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Schedule.op unresolved from ref:Schedule.layer:Layer %s > tsu.unit:333, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApLayer[st.Klayerp].MyName
		v, _ = st.Names["op"]
		err, res = fnd3(act, strconv.Itoa(st.Klayerp) + "_Op_" + v, v, "ref_child:Schedule.op:Layer." + parent + "." + v + " from ref:Schedule.layer", "*", st.LineNo, "tsu.unit:333, g_runh.act:236")
		st.Kopp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApBuildRule {

//  tsu.unit:446, g_runh.act:262
	p = st.Kparentp
	if p >= 0 {
		st.Kprojectp = p
	} else if "*" != "*" {
		fmt.Printf("ref_copy:BuildRule.project unresolved from word:BuildRule.template:..x %s (*) > tsu.unit:446, g_runh.act:271\n", st.LineNo)
		errs += 1
	}
//  tsu.unit:447, g_runh.act:224

 
	if st.Kprojectp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:BuildRule.validate_against unresolved from up_copy:BuildRule.project:Project %s > tsu.unit:447, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApProject[st.Kprojectp].MyName
		v, _ = st.Names["validate_against"]
		err, res = fnd3(act, strconv.Itoa(st.Kprojectp) + "_TargetConfig_" + v, v, "ref_child:BuildRule.validate_against:Project." + parent + "." + v + " from up_copy:BuildRule.project", "*", st.LineNo, "tsu.unit:447, g_runh.act:236")
		st.Kvalidate_againstp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApAutoTuneConfig {

//  tsu-auto.unit:169, g_runh.act:224

 
	if st.Kopt_fromp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:AutoTuneConfig.resume_from unresolved from ref:AutoTuneConfig.opt_from:OptimizationRun %s > tsu-auto.unit:169, g_runh.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApOptimizationRun[st.Kopt_fromp].MyName
		v, _ = st.Names["resume_from"]
		err, res = fnd3(act, strconv.Itoa(st.Kopt_fromp) + "_Checkpoint_" + v, v, "ref_child:AutoTuneConfig.resume_from:OptimizationRun." + parent + "." + v + " from ref:AutoTuneConfig.opt_from", "*", st.LineNo, "tsu-auto.unit:169, g_runh.act:236")
		st.Kresume_fromp = res
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
	if va[0] == "TSU" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TSU_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTSU[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTSU[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTSU {
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
	if va[0] == "TSUKernel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TSUKernel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTSUKernel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTSUKernel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTSUKernel {
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
	if va[0] == "THRML" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["THRML_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTHRML[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTHRML[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTHRML {
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
	if va[0] == "PGMSchema" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PGMSchema_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPGMSchema[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPGMSchema[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPGMSchema {
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
	if va[0] == "TSUSamplingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TSUSamplingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTSUSamplingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTSUSamplingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTSUSamplingOp {
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
	if va[0] == "BlockGibbsOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["BlockGibbsOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBlockGibbsOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBlockGibbsOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBlockGibbsOp {
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
	if va[0] == "EnergyFunction" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnergyFunction_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnergyFunction[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnergyFunction[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnergyFunction {
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
	if va[0] == "EnergyFactor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnergyFactor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnergyFactor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnergyFactor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnergyFactor {
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
	if va[0] == "TSUCompilation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TSUCompilation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTSUCompilation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTSUCompilation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTSUCompilation {
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
	if va[0] == "TSUSubstrateModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TSUSubstrateModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTSUSubstrateModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTSUSubstrateModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTSUSubstrateModel {
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
	if va[0] == "ThermodynamicSimulation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ThermodynamicSimulation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApThermodynamicSimulation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApThermodynamicSimulation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApThermodynamicSimulation {
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
	if va[0] == "PhysicalConstraint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PhysicalConstraint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPhysicalConstraint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPhysicalConstraint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPhysicalConstraint {
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
	if va[0] == "SamplingAlgorithmRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SamplingAlgorithmRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSamplingAlgorithmRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSamplingAlgorithmRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSamplingAlgorithmRule {
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
	if va[0] == "BackendRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["BackendRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBackendRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBackendRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBackendRule {
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
	if va[0] == "HardwareRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HardwareRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHardwareRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHardwareRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHardwareRule {
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
	if va[0] == "ScheduleParameter" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ScheduleParameter_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApScheduleParameter[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApScheduleParameter[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApScheduleParameter {
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
	if va[0] == "EvolutionStrategy" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EvolutionStrategy_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEvolutionStrategy[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEvolutionStrategy[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEvolutionStrategy {
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
	if va[0] == "PerformanceMetric" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PerformanceMetric_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPerformanceMetric[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPerformanceMetric[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPerformanceMetric {
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
	if va[0] == "ExperimentTrial" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ExperimentTrial_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApExperimentTrial[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApExperimentTrial[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApExperimentTrial {
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
	if va[0] == "TrialMetrics" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TrialMetrics_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTrialMetrics[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTrialMetrics[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTrialMetrics {
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
	if va[0] == "HyperParameterRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HyperParameterRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHyperParameterRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHyperParameterRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHyperParameterRule {
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
	if va[0] == "MetaLearning" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MetaLearning_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMetaLearning[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMetaLearning[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMetaLearning {
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
	if va[0] == "AutoTuneConfig" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AutoTuneConfig_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAutoTuneConfig[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAutoTuneConfig[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAutoTuneConfig {
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
	if va[0] == "SensitivityAnalysis" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["SensitivityAnalysis_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApSensitivityAnalysis[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApSensitivityAnalysis[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApSensitivityAnalysis {
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
	if va[0] == "EnsembleStrategy" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["EnsembleStrategy_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApEnsembleStrategy[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApEnsembleStrategy[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApEnsembleStrategy {
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
	if va[0] == "OptimizationPattern" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["OptimizationPattern_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApOptimizationPattern[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApOptimizationPattern[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApOptimizationPattern {
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
	if va[0] == "TransferLearning" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["TransferLearning_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApTransferLearning[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApTransferLearning[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApTransferLearning {
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
	if va[0] == "MultiObjective" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiObjective_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiObjective[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiObjective[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiObjective {
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
	if va[0] == "ParetoFront" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ParetoFront_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApParetoFront[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApParetoFront[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApParetoFront {
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
	if va[0] == "DiagnosticRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DiagnosticRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDiagnosticRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDiagnosticRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDiagnosticRule {
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
	if va[0] == "FailureMode" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["FailureMode_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApFailureMode[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApFailureMode[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApFailureMode {
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
	if va[0] == "ResourceBudget" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ResourceBudget_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApResourceBudget[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApResourceBudget[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApResourceBudget {
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
	if va[0] == "AdaptiveScheduling" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AdaptiveScheduling_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAdaptiveScheduling[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAdaptiveScheduling[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAdaptiveScheduling {
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
	if va[0] == "MultiStateModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiStateModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiStateModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiStateModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiStateModel {
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
	if va[0] == "LoopyPGM" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["LoopyPGM_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApLoopyPGM[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApLoopyPGM[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApLoopyPGM {
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
	if va[0] == "VariationalModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["VariationalModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApVariationalModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApVariationalModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApVariationalModel {
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
	if va[0] == "HybridSampling" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HybridSampling_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHybridSampling[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHybridSampling[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHybridSampling {
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
	if va[0] == "PottsOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PottsOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPottsOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPottsOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPottsOp {
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
	if va[0] == "PottsKernel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["PottsKernel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApPottsKernel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApPottsKernel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApPottsKernel {
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
	if va[0] == "MultiStateOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiStateOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiStateOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiStateOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiStateOp {
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
	if va[0] == "ClusterSamplingOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ClusterSamplingOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApClusterSamplingOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApClusterSamplingOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApClusterSamplingOp {
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
	if va[0] == "VariationalOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["VariationalOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApVariationalOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApVariationalOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApVariationalOp {
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
	if va[0] == "BeliefPropagationOp" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["BeliefPropagationOp_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApBeliefPropagationOp[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApBeliefPropagationOp[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApBeliefPropagationOp {
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
	if va[0] == "MultiStateEnergyFactor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiStateEnergyFactor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiStateEnergyFactor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiStateEnergyFactor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiStateEnergyFactor {
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
	if va[0] == "VariationalFactor" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["VariationalFactor_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApVariationalFactor[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApVariationalFactor[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApVariationalFactor {
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
	if va[0] == "DegenerateEnergy" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DegenerateEnergy_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDegenerateEnergy[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDegenerateEnergy[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDegenerateEnergy {
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
	if va[0] == "MultiLevelTSU" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiLevelTSU_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiLevelTSU[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiLevelTSU[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiLevelTSU {
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
	if va[0] == "AdvancedTSUCompilation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AdvancedTSUCompilation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAdvancedTSUCompilation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAdvancedTSUCompilation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAdvancedTSUCompilation {
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
	if va[0] == "NoiseModel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["NoiseModel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApNoiseModel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApNoiseModel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApNoiseModel {
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
	if va[0] == "HybridConfig" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["HybridConfig_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApHybridConfig[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApHybridConfig[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApHybridConfig {
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
	if va[0] == "ELBOTracking" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ELBOTracking_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApELBOTracking[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApELBOTracking[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApELBOTracking {
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
	if va[0] == "ErgodicityValidation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ErgodicityValidation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApErgodicityValidation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApErgodicityValidation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApErgodicityValidation {
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
	if va[0] == "MultiStateConstraint" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiStateConstraint_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiStateConstraint[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiStateConstraint[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiStateConstraint {
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
	if va[0] == "DegeneracySimulation" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["DegeneracySimulation_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDegeneracySimulation[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDegeneracySimulation[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDegeneracySimulation {
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
	if va[0] == "ClusterKernel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["ClusterKernel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApClusterKernel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApClusterKernel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApClusterKernel {
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
	if va[0] == "VariationalKernel" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["VariationalKernel_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApVariationalKernel[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApVariationalKernel[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApVariationalKernel {
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
	if va[0] == "MultiStateDistributionRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiStateDistributionRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiStateDistributionRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiStateDistributionRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiStateDistributionRule {
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
	if va[0] == "AdvancedSamplingAlgorithmRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AdvancedSamplingAlgorithmRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAdvancedSamplingAlgorithmRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAdvancedSamplingAlgorithmRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAdvancedSamplingAlgorithmRule {
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
	if va[0] == "VIAlgorithmRule" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["VIAlgorithmRule_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApVIAlgorithmRule[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApVIAlgorithmRule[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApVIAlgorithmRule {
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
	if va[0] == "MultiStateProject" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["MultiStateProject_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApMultiStateProject[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApMultiStateProject[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApMultiStateProject {
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
	if va[0] == "AutoTuneExtension" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["AutoTuneExtension_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApAutoTuneExtension[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApAutoTuneExtension[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApAutoTuneExtension {
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
	if tok == "TSU" { errs += loadTSU(act,ln,pos,lno,flag,nm) }
	if tok == "Kernel" { errs += loadKernel(act,ln,pos,lno,flag,nm) }
	if tok == "TSUKernel" { errs += loadTSUKernel(act,ln,pos,lno,flag,nm) }
	if tok == "KernelParam" { errs += loadKernelParam(act,ln,pos,lno,flag,nm) }
	if tok == "KernelOp" { errs += loadKernelOp(act,ln,pos,lno,flag,nm) }
	if tok == "Optimization" { errs += loadOptimization(act,ln,pos,lno,flag,nm) }
	if tok == "Fusion" { errs += loadFusion(act,ln,pos,lno,flag,nm) }
	if tok == "Framework" { errs += loadFramework(act,ln,pos,lno,flag,nm) }
	if tok == "THRML" { errs += loadTHRML(act,ln,pos,lno,flag,nm) }
	if tok == "PGMSchema" { errs += loadPGMSchema(act,ln,pos,lno,flag,nm) }
	if tok == "Model" { errs += loadModel(act,ln,pos,lno,flag,nm) }
	if tok == "Tensor" { errs += loadTensor(act,ln,pos,lno,flag,nm) }
	if tok == "Layer" { errs += loadLayer(act,ln,pos,lno,flag,nm) }
	if tok == "Op" { errs += loadOp(act,ln,pos,lno,flag,nm) }
	if tok == "Arg" { errs += loadArg(act,ln,pos,lno,flag,nm) }
	if tok == "SamplingOp" { errs += loadSamplingOp(act,ln,pos,lno,flag,nm) }
	if tok == "TSUSamplingOp" { errs += loadTSUSamplingOp(act,ln,pos,lno,flag,nm) }
	if tok == "BlockGibbsOp" { errs += loadBlockGibbsOp(act,ln,pos,lno,flag,nm) }
	if tok == "EnergyFunction" { errs += loadEnergyFunction(act,ln,pos,lno,flag,nm) }
	if tok == "EnergyFactor" { errs += loadEnergyFactor(act,ln,pos,lno,flag,nm) }
	if tok == "TSUCompilation" { errs += loadTSUCompilation(act,ln,pos,lno,flag,nm) }
	if tok == "TSUSubstrateModel" { errs += loadTSUSubstrateModel(act,ln,pos,lno,flag,nm) }
	if tok == "Config" { errs += loadConfig(act,ln,pos,lno,flag,nm) }
	if tok == "Schedule" { errs += loadSchedule(act,ln,pos,lno,flag,nm) }
	if tok == "ThermodynamicSimulation" { errs += loadThermodynamicSimulation(act,ln,pos,lno,flag,nm) }
	if tok == "Validation" { errs += loadValidation(act,ln,pos,lno,flag,nm) }
	if tok == "PhysicalConstraint" { errs += loadPhysicalConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "DistributionRule" { errs += loadDistributionRule(act,ln,pos,lno,flag,nm) }
	if tok == "SamplingAlgorithmRule" { errs += loadSamplingAlgorithmRule(act,ln,pos,lno,flag,nm) }
	if tok == "BackendRule" { errs += loadBackendRule(act,ln,pos,lno,flag,nm) }
	if tok == "HardwareRule" { errs += loadHardwareRule(act,ln,pos,lno,flag,nm) }
	if tok == "Project" { errs += loadProject(act,ln,pos,lno,flag,nm) }
	if tok == "TargetConfig" { errs += loadTargetConfig(act,ln,pos,lno,flag,nm) }
	if tok == "BuildRule" { errs += loadBuildRule(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpace" { errs += loadSearchSpace(act,ln,pos,lno,flag,nm) }
	if tok == "SearchParameter" { errs += loadSearchParameter(act,ln,pos,lno,flag,nm) }
	if tok == "ScheduleParameter" { errs += loadScheduleParameter(act,ln,pos,lno,flag,nm) }
	if tok == "EvolutionStrategy" { errs += loadEvolutionStrategy(act,ln,pos,lno,flag,nm) }
	if tok == "FitnessFunction" { errs += loadFitnessFunction(act,ln,pos,lno,flag,nm) }
	if tok == "AdaptiveParameter" { errs += loadAdaptiveParameter(act,ln,pos,lno,flag,nm) }
	if tok == "PerformanceMetric" { errs += loadPerformanceMetric(act,ln,pos,lno,flag,nm) }
	if tok == "ExperimentTrial" { errs += loadExperimentTrial(act,ln,pos,lno,flag,nm) }
	if tok == "TrialMetrics" { errs += loadTrialMetrics(act,ln,pos,lno,flag,nm) }
	if tok == "OptimizationRun" { errs += loadOptimizationRun(act,ln,pos,lno,flag,nm) }
	if tok == "Checkpoint" { errs += loadCheckpoint(act,ln,pos,lno,flag,nm) }
	if tok == "HyperParameterRule" { errs += loadHyperParameterRule(act,ln,pos,lno,flag,nm) }
	if tok == "MetaLearning" { errs += loadMetaLearning(act,ln,pos,lno,flag,nm) }
	if tok == "AutoTuneConfig" { errs += loadAutoTuneConfig(act,ln,pos,lno,flag,nm) }
	if tok == "SensitivityAnalysis" { errs += loadSensitivityAnalysis(act,ln,pos,lno,flag,nm) }
	if tok == "EnsembleStrategy" { errs += loadEnsembleStrategy(act,ln,pos,lno,flag,nm) }
	if tok == "OptimizationPattern" { errs += loadOptimizationPattern(act,ln,pos,lno,flag,nm) }
	if tok == "TransferLearning" { errs += loadTransferLearning(act,ln,pos,lno,flag,nm) }
	if tok == "MultiObjective" { errs += loadMultiObjective(act,ln,pos,lno,flag,nm) }
	if tok == "ParetoFront" { errs += loadParetoFront(act,ln,pos,lno,flag,nm) }
	if tok == "DiagnosticRule" { errs += loadDiagnosticRule(act,ln,pos,lno,flag,nm) }
	if tok == "FailureMode" { errs += loadFailureMode(act,ln,pos,lno,flag,nm) }
	if tok == "ResourceBudget" { errs += loadResourceBudget(act,ln,pos,lno,flag,nm) }
	if tok == "AdaptiveScheduling" { errs += loadAdaptiveScheduling(act,ln,pos,lno,flag,nm) }
	if tok == "MultiStateModel" { errs += loadMultiStateModel(act,ln,pos,lno,flag,nm) }
	if tok == "LoopyPGM" { errs += loadLoopyPGM(act,ln,pos,lno,flag,nm) }
	if tok == "VariationalModel" { errs += loadVariationalModel(act,ln,pos,lno,flag,nm) }
	if tok == "HybridSampling" { errs += loadHybridSampling(act,ln,pos,lno,flag,nm) }
	if tok == "PottsOp" { errs += loadPottsOp(act,ln,pos,lno,flag,nm) }
	if tok == "PottsKernel" { errs += loadPottsKernel(act,ln,pos,lno,flag,nm) }
	if tok == "MultiStateOp" { errs += loadMultiStateOp(act,ln,pos,lno,flag,nm) }
	if tok == "ClusterSamplingOp" { errs += loadClusterSamplingOp(act,ln,pos,lno,flag,nm) }
	if tok == "VariationalOp" { errs += loadVariationalOp(act,ln,pos,lno,flag,nm) }
	if tok == "BeliefPropagationOp" { errs += loadBeliefPropagationOp(act,ln,pos,lno,flag,nm) }
	if tok == "MultiStateEnergyFactor" { errs += loadMultiStateEnergyFactor(act,ln,pos,lno,flag,nm) }
	if tok == "VariationalFactor" { errs += loadVariationalFactor(act,ln,pos,lno,flag,nm) }
	if tok == "DegenerateEnergy" { errs += loadDegenerateEnergy(act,ln,pos,lno,flag,nm) }
	if tok == "MultiLevelTSU" { errs += loadMultiLevelTSU(act,ln,pos,lno,flag,nm) }
	if tok == "AdvancedTSUCompilation" { errs += loadAdvancedTSUCompilation(act,ln,pos,lno,flag,nm) }
	if tok == "NoiseModel" { errs += loadNoiseModel(act,ln,pos,lno,flag,nm) }
	if tok == "HybridConfig" { errs += loadHybridConfig(act,ln,pos,lno,flag,nm) }
	if tok == "ELBOTracking" { errs += loadELBOTracking(act,ln,pos,lno,flag,nm) }
	if tok == "ErgodicityValidation" { errs += loadErgodicityValidation(act,ln,pos,lno,flag,nm) }
	if tok == "MultiStateConstraint" { errs += loadMultiStateConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "DegeneracySimulation" { errs += loadDegeneracySimulation(act,ln,pos,lno,flag,nm) }
	if tok == "ClusterKernel" { errs += loadClusterKernel(act,ln,pos,lno,flag,nm) }
	if tok == "VariationalKernel" { errs += loadVariationalKernel(act,ln,pos,lno,flag,nm) }
	if tok == "MultiStateDistributionRule" { errs += loadMultiStateDistributionRule(act,ln,pos,lno,flag,nm) }
	if tok == "AdvancedSamplingAlgorithmRule" { errs += loadAdvancedSamplingAlgorithmRule(act,ln,pos,lno,flag,nm) }
	if tok == "VIAlgorithmRule" { errs += loadVIAlgorithmRule(act,ln,pos,lno,flag,nm) }
	if tok == "MultiStateProject" { errs += loadMultiStateProject(act,ln,pos,lno,flag,nm) }
	if tok == "AutoTuneExtension" { errs += loadAutoTuneExtension(act,ln,pos,lno,flag,nm) }
	return errs
}

