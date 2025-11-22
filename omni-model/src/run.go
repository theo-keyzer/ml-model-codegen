package main

import (
	"strings"
	"fmt"
	"strconv"
)

type ActT struct {
	index       map[string]int
	ApProject [] *KpProject
	ApDomain [] *KpDomain
	ApHardware [] *KpHardware
	ApModel [] *KpModel
	ApLayer [] *KpLayer
	ApBlock [] *KpBlock
	ApTensor [] *KpTensor
	ApOp [] *KpOp
	ApArg [] *KpArg
	ApConfig [] *KpConfig
	ApKernel [] *KpKernel
	ApEnergyFunction [] *KpEnergyFunction
	ApSearchSpace [] *KpSearchSpace
	ApDimension [] *KpDimension
	ApStrategy [] *KpStrategy
	ApConstraint [] *KpConstraint
	ApMetric [] *KpMetric
	ApCheckpoint [] *KpCheckpoint
	ApFusion [] *KpFusion
	ApControlFlow [] *KpControlFlow
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
	for _, st := range act.ApProject {

//  omni.unit:13, g_runa.act:180

		v, _ = st.Names["domain"].(string)
		err, res = fnd3(act, "Domain_" + v, v, "ref:Project.domain:Domain." + v,  "*", st.LineNo, "omni.unit:13, g_runa.act:184" );
		st.Kdomainp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:14, g_runa.act:180

		v, _ = st.Names["model"].(string)
		err, res = fnd3(act, "Model_" + v, v, "ref:Project.model:Model." + v,  "+", st.LineNo, "omni.unit:14, g_runa.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:15, g_runa.act:180

		v, _ = st.Names["strategy"].(string)
		err, res = fnd3(act, "Strategy_" + v, v, "ref:Project.strategy:Strategy." + v,  "*", st.LineNo, "omni.unit:15, g_runa.act:184" );
		st.Kstrategyp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:16, g_runa.act:180

		v, _ = st.Names["hardware"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Project.hardware:Hardware." + v,  "*", st.LineNo, "omni.unit:16, g_runa.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApHardware {

//  omni.unit:31, g_runa.act:180

		v, _ = st.Names["parent_hw"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Hardware.parent_hw:Hardware." + v,  "*", st.LineNo, "omni.unit:31, g_runa.act:184" );
		st.Kparent_hwp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:32, g_runa.act:180

		v, _ = st.Names["emulation"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Hardware.emulation:Hardware." + v,  "*", st.LineNo, "omni.unit:32, g_runa.act:184" );
		st.Kemulationp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:33, g_runa.act:180

		v, _ = st.Names["noise_model"].(string)
		err, res = fnd3(act, "Constraint_" + v, v, "ref:Hardware.noise_model:Constraint." + v,  "*", st.LineNo, "omni.unit:33, g_runa.act:184" );
		st.Knoise_modelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApModel {

//  omni.unit:41, g_runa.act:180

		v, _ = st.Names["hardware"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Model.hardware:Hardware." + v,  "*", st.LineNo, "omni.unit:41, g_runa.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:42, g_runa.act:180

		v, _ = st.Names["search_space"].(string)
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:Model.search_space:SearchSpace." + v,  "*", st.LineNo, "omni.unit:42, g_runa.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:43, g_runa.act:180

		v, _ = st.Names["config"].(string)
		err, res = fnd3(act, "Config_" + v, v, "ref:Model.config:Config." + v,  "*", st.LineNo, "omni.unit:43, g_runa.act:184" );
		st.Kconfigp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApLayer {

//  omni.unit:52, g_runa.act:209

		v, _ = st.Names["parent_layer"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Layer_" + v,v, "ref_link:Layer.parent_layer:Model." + st.Parent + ".Layer." + v,  "*", st.LineNo, "omni.unit:52, g_runa.act:212" );
		st.Kparent_layerp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApBlock {

//  omni.unit:60, g_runa.act:180

		v, _ = st.Names["model"].(string)
		err, res = fnd3(act, "Model_" + v, v, "ref:Block.model:Model." + v,  "*", st.LineNo, "omni.unit:60, g_runa.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApTensor {

//  omni.unit:68, g_runa.act:209

		v, _ = st.Names["producer"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Op_" + v,v, "ref_link:Tensor.producer:Model." + st.Parent + ".Op." + v,  "*", st.LineNo, "omni.unit:68, g_runa.act:212" );
		st.Kproducerp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:69, g_runa.act:180

		v, _ = st.Names["distribution"].(string)
		err, res = fnd3(act, "EnergyFunction_" + v, v, "ref:Tensor.distribution:EnergyFunction." + v,  "*", st.LineNo, "omni.unit:69, g_runa.act:184" );
		st.Kdistributionp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApOp {

//  omni.unit:79, g_runa.act:180

		v, _ = st.Names["hardware"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Op.hardware:Hardware." + v,  "*", st.LineNo, "omni.unit:79, g_runa.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:80, g_runa.act:180

		v, _ = st.Names["energy_fn"].(string)
		err, res = fnd3(act, "EnergyFunction_" + v, v, "ref:Op.energy_fn:EnergyFunction." + v,  "*", st.LineNo, "omni.unit:80, g_runa.act:184" );
		st.Kenergy_fnp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:81, g_runa.act:180

		v, _ = st.Names["search_space"].(string)
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:Op.search_space:SearchSpace." + v,  "*", st.LineNo, "omni.unit:81, g_runa.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:82, g_runa.act:180

		v, _ = st.Names["strategy"].(string)
		err, res = fnd3(act, "Strategy_" + v, v, "ref:Op.strategy:Strategy." + v,  "*", st.LineNo, "omni.unit:82, g_runa.act:184" );
		st.Kstrategyp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:85, g_runa.act:209

		v, _ = st.Names["predicate"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Tensor_" + v,v, "ref_link:Op.predicate:Model." + st.Parent + ".Tensor." + v,  "*", st.LineNo, "omni.unit:85, g_runa.act:212" );
		st.Kpredicatep = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:86, g_runa.act:209

		v, _ = st.Names["next_op"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Op_" + v,v, "ref_link:Op.next_op:Model." + st.Parent + ".Op." + v,  "*", st.LineNo, "omni.unit:86, g_runa.act:212" );
		st.Knext_opp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:87, g_runa.act:209

		v, _ = st.Names["layer"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kparentp) + "_Layer_" + v,v, "ref_link:Op.layer:Model." + st.Parent + ".Layer." + v,  "*", st.LineNo, "omni.unit:87, g_runa.act:212" );
		st.Klayerp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApConfig {

//  omni.unit:112, g_runa.act:180

		v, _ = st.Names["schedule"].(string)
		err, res = fnd3(act, "Strategy_" + v, v, "ref:Config.schedule:Strategy." + v,  "*", st.LineNo, "omni.unit:112, g_runa.act:184" );
		st.Kschedulep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApKernel {

//  omni.unit:121, g_runa.act:180

		v, _ = st.Names["hardware"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Kernel.hardware:Hardware." + v,  "*", st.LineNo, "omni.unit:121, g_runa.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApSearchSpace {

//  omni.unit:137, g_runa.act:180

		v, _ = st.Names["target_model"].(string)
		err, res = fnd3(act, "Model_" + v, v, "ref:SearchSpace.target_model:Model." + v,  "*", st.LineNo, "omni.unit:137, g_runa.act:184" );
		st.Ktarget_modelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApStrategy {

//  omni.unit:152, g_runa.act:180

		v, _ = st.Names["search_space"].(string)
		err, res = fnd3(act, "SearchSpace_" + v, v, "ref:Strategy.search_space:SearchSpace." + v,  "*", st.LineNo, "omni.unit:152, g_runa.act:184" );
		st.Ksearch_spacep = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:153, g_runa.act:180

		v, _ = st.Names["fitness"].(string)
		err, res = fnd3(act, "Metric_" + v, v, "ref:Strategy.fitness:Metric." + v,  "*", st.LineNo, "omni.unit:153, g_runa.act:184" );
		st.Kfitnessp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApConstraint {

//  omni.unit:162, g_runa.act:180

		v, _ = st.Names["target_hw"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Constraint.target_hw:Hardware." + v,  "*", st.LineNo, "omni.unit:162, g_runa.act:184" );
		st.Ktarget_hwp = res
		if (err == false) {
			errs += 1
		}
//  omni.unit:163, g_runa.act:180

		v, _ = st.Names["model"].(string)
		err, res = fnd3(act, "Model_" + v, v, "ref:Constraint.model:Model." + v,  "*", st.LineNo, "omni.unit:163, g_runa.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApMetric {

//  omni.unit:172, g_runa.act:180

		v, _ = st.Names["target_hw"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Metric.target_hw:Hardware." + v,  "*", st.LineNo, "omni.unit:172, g_runa.act:184" );
		st.Ktarget_hwp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApCheckpoint {

//  omni.unit:180, g_runa.act:180

		v, _ = st.Names["model"].(string)
		err, res = fnd3(act, "Model_" + v, v, "ref:Checkpoint.model:Model." + v,  "*", st.LineNo, "omni.unit:180, g_runa.act:184" );
		st.Kmodelp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApFusion {

//  omni.unit:188, g_runa.act:180

		v, _ = st.Names["hardware"].(string)
		err, res = fnd3(act, "Hardware_" + v, v, "ref:Fusion.hardware:Hardware." + v,  "*", st.LineNo, "omni.unit:188, g_runa.act:184" );
		st.Khardwarep = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApAll {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:34, g_runa.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApDu {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:46, g_runa.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApIts {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:87, g_runa.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApThis {

		err, res = fnd2(act, "Actor_" + st.Kactor, st.Kactor,  ".", st.LineNo, "act.unit:186, g_runa.act:170" );
		st.Kactorp = res
		if (err == false) {
			errs += 1
		}
	}
	for _, st := range act.ApArg {

//  omni.unit:104, g_runa.act:262
	p = st.Me
	p = act.ApArg[p].Kparentp
	p = act.ApOp[p].Kparentp
	if p >= 0 {
		st.Kmodelp = p
	} else if "*" != "*" {
		fmt.Printf("ref_copy:Arg.model unresolved from word:Arg.role:..x %s (*) > omni.unit:104, g_runa.act:275\n", st.LineNo)
		errs += 1
	}
//  omni.unit:105, g_runa.act:224

 
	if st.Kmodelp < 0 {
		if "+" != "*" {
			fmt.Printf("ref_child:Arg.tensor unresolved from up_copy:Arg.model:Model %s > omni.unit:105, g_runa.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["tensor"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Tensor_" + v, v, "ref_child:Arg.tensor:Model." + parent + "." + v + " from up_copy:Arg.model", "+", st.LineNo, "omni.unit:105, g_runa.act:236")
		st.Ktensorp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApConstraint {

//  omni.unit:164, g_runa.act:224

 
	if st.Kmodelp < 0 {
		if "*" != "*" {
			fmt.Printf("ref_child:Constraint.target_op unresolved from ref:Constraint.model:Model %s > omni.unit:164, g_runa.act:229", st.LineNo)
			errs += 1
		}
	} else {
		parent := act.ApModel[st.Kmodelp].MyName
		v, _ = st.Names["target_op"].(string)
		err, res = fnd3(act, strconv.Itoa(st.Kmodelp) + "_Op_" + v, v, "ref_child:Constraint.target_op:Model." + parent + "." + v + " from ref:Constraint.model", "*", st.LineNo, "omni.unit:164, g_runa.act:236")
		st.Ktarget_opp = res
		if !err {
			errs += 1
		}
	}
	}
	for _, st := range act.ApControlFlow {

//  omni.unit:196, g_runa.act:262
	p = st.Me
	p = act.ApControlFlow[p].Kparentp
	p = act.ApOp[p].Kparentp
	if p >= 0 {
		st.Kmodel1p = p
	} else if "*" != "*" {
		fmt.Printf("ref_copy:ControlFlow.model1 unresolved from key:ControlFlow.control:..x %s (*) > omni.unit:196, g_runa.act:275\n", st.LineNo)
		errs += 1
	}
//  omni.unit:198, g_runa.act:262
	p = st.Me
	p = act.ApControlFlow[p].Kparentp
	p = act.ApOp[p].Kparentp
	if p >= 0 {
		st.Kmodel2p = p
	} else if "*" != "*" {
		fmt.Printf("ref_copy:ControlFlow.model2 unresolved from ref_cild:ControlFlow.predicate:Tensor.x %s (*) > omni.unit:198, g_runa.act:275\n", st.LineNo)
		errs += 1
	}
//  omni.unit:200, g_runa.act:262
	p = st.Me
	p = act.ApControlFlow[p].Kparentp
	p = act.ApOp[p].Kparentp
	if p >= 0 {
		st.Kmodel3p = p
	} else if "*" != "*" {
		fmt.Printf("ref_copy:ControlFlow.model3 unresolved from ref_cild:ControlFlow.branch_true:Op.x %s (*) > omni.unit:200, g_runa.act:275\n", st.LineNo)
		errs += 1
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
	if va[0] == "Dimension" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Dimension_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApDimension[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApDimension[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApDimension {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if ret != 0 {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Strategy" {
		if (len(va) > 1 && len(va[1]) > 0) {
			en, er := glob.Dats.index["Strategy_" + va[1] ];
			if !er {
				if len(va) > 2 {
					return( glob.Dats.ApStrategy[en].DoIts(glob, va[2:], lno) )
				}
				return( GoAct(glob, glob.Dats.ApStrategy[en]) )
			}
			return(0)
		}
		for _, st := range glob.Dats.ApStrategy {
			if len(va) > 2 {
				ret := st.DoIts(glob, va[2:], lno)
				if ret != 0 {
					return(ret)
				}
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
	fmt.Printf("?No all %s cmd ?%s? > g_runa.act:43", va[0], lno);
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

func Loadh(act *ActT, toks string, ln string, pos int, lno string, nm map[string]any) int {
	errs := 0
	ss := strings.Split(toks,".")
	tok := ss[0]
	flag := ss[1:]
	if tok == "Project" { errs += loadProject(act,ln,pos,lno,flag,nm) }
	if tok == "Domain" { errs += loadDomain(act,ln,pos,lno,flag,nm) }
	if tok == "Hardware" { errs += loadHardware(act,ln,pos,lno,flag,nm) }
	if tok == "Model" { errs += loadModel(act,ln,pos,lno,flag,nm) }
	if tok == "Layer" { errs += loadLayer(act,ln,pos,lno,flag,nm) }
	if tok == "Block" { errs += loadBlock(act,ln,pos,lno,flag,nm) }
	if tok == "Tensor" { errs += loadTensor(act,ln,pos,lno,flag,nm) }
	if tok == "Op" { errs += loadOp(act,ln,pos,lno,flag,nm) }
	if tok == "Arg" { errs += loadArg(act,ln,pos,lno,flag,nm) }
	if tok == "Config" { errs += loadConfig(act,ln,pos,lno,flag,nm) }
	if tok == "Kernel" { errs += loadKernel(act,ln,pos,lno,flag,nm) }
	if tok == "EnergyFunction" { errs += loadEnergyFunction(act,ln,pos,lno,flag,nm) }
	if tok == "SearchSpace" { errs += loadSearchSpace(act,ln,pos,lno,flag,nm) }
	if tok == "Dimension" { errs += loadDimension(act,ln,pos,lno,flag,nm) }
	if tok == "Strategy" { errs += loadStrategy(act,ln,pos,lno,flag,nm) }
	if tok == "Constraint" { errs += loadConstraint(act,ln,pos,lno,flag,nm) }
	if tok == "Metric" { errs += loadMetric(act,ln,pos,lno,flag,nm) }
	if tok == "Checkpoint" { errs += loadCheckpoint(act,ln,pos,lno,flag,nm) }
	if tok == "Fusion" { errs += loadFusion(act,ln,pos,lno,flag,nm) }
	if tok == "ControlFlow" { errs += loadControlFlow(act,ln,pos,lno,flag,nm) }
	return errs
}

