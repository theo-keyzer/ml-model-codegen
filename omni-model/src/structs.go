package main

import (
	"fmt"
	"strconv"
	"encoding/json"
	"maps"
)

type Kp interface {
	DoIts(glob *GlobT, va []string, lno string) int
	GetVar(glob *GlobT, va []string, lno string) (bool, string)
	GetLineNo() string
	TypeName() string
}

type KpProject struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kdomainp int
	Kmodelp int
	Kstrategyp int
	Khardwarep int
}

func (me KpProject) TypeName() string {
    return me.Comp
}
func (me KpProject) GetLineNo() string {
	return me.LineNo
}

func loadProject(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpProject)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProject)
	st.LineNo = lno
	st.Comp = "Project";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kdomainp = -1
	st.Kmodelp = -1
	st.Kstrategyp = -1
	st.Khardwarep = -1
	name,_ := st.Names["project"].(string)
	st.Names["_key"] = "project"
	act.index["Project_" + name] = st.Me;
	st.MyName = name
	act.ApProject = append(act.ApProject, st)
	return 0
}

func (me KpProject) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "domain" { // omni.unit:13, g_structa.act:623
		if (me.Kdomainp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Kdomainp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model" { // omni.unit:14, g_structa.act:623
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "strategy" { // omni.unit:15, g_structa.act:623
		if (me.Kstrategyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApStrategy[ me.Kstrategyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // omni.unit:16, g_structa.act:623
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // omni.unit:8, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Project > omni.unit:8, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Project > omni.unit:8, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpProject) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "domain" {
		if me.Kdomainp >= 0 {
			st := glob.Dats.ApDomain[ me.Kdomainp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "model" {
		if me.Kmodelp >= 0 {
			st := glob.Dats.ApModel[ me.Kmodelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "strategy" {
		if me.Kstrategyp >= 0 {
			st := glob.Dats.ApStrategy[ me.Kstrategyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "hardware" {
		if me.Khardwarep >= 0 {
			st := glob.Dats.ApHardware[ me.Khardwarep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Project %s,%s > omni.unit:8, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDomain struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
}

func (me KpDomain) TypeName() string {
    return me.Comp
}
func (me KpDomain) GetLineNo() string {
	return me.LineNo
}

func loadDomain(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpDomain)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDomain)
	st.LineNo = lno
	st.Comp = "Domain";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["name"].(string)
	st.Names["_key"] = "name"
	act.index["Domain_" + name] = st.Me;
	st.MyName = name
	act.ApDomain = append(act.ApDomain, st)
	return 0
}

func (me KpDomain) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Project_domain" && len(va) > 1) { // omni.unit:13, g_structa.act:712
		for _, st := range glob.Dats.ApProject {
			if (st.Kdomainp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:19, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Domain > omni.unit:19, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Domain > omni.unit:19, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpDomain) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Project_domain") { // omni.unit:13, g_structa.act:597
		for _, st := range glob.Dats.ApProject {
			if (st.Kdomainp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Domain %s,%s > omni.unit:19, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHardware struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparent_hwp int
	Kemulationp int
	Knoise_modelp int
}

func (me KpHardware) TypeName() string {
    return me.Comp
}
func (me KpHardware) GetLineNo() string {
	return me.LineNo
}

func loadHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHardware)
	st.LineNo = lno
	st.Comp = "Hardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparent_hwp = -1
	st.Kemulationp = -1
	st.Knoise_modelp = -1
	name,_ := st.Names["hardware"].(string)
	st.Names["_key"] = "hardware"
	act.index["Hardware_" + name] = st.Me;
	st.MyName = name
	act.ApHardware = append(act.ApHardware, st)
	return 0
}

func (me KpHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "parent_hw" { // omni.unit:31, g_structa.act:623
		if (me.Kparent_hwp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparent_hwp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "emulation" { // omni.unit:32, g_structa.act:623
		if (me.Kemulationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kemulationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "noise_model" { // omni.unit:33, g_structa.act:623
		if (me.Knoise_modelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConstraint[ me.Knoise_modelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Project_hardware" && len(va) > 1) { // omni.unit:16, g_structa.act:712
		for _, st := range glob.Dats.ApProject {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Hardware_parent_hw" && len(va) > 1) { // omni.unit:31, g_structa.act:712
		for _, st := range glob.Dats.ApHardware {
			if (st.Kparent_hwp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Hardware_emulation" && len(va) > 1) { // omni.unit:32, g_structa.act:712
		for _, st := range glob.Dats.ApHardware {
			if (st.Kemulationp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Model_hardware" && len(va) > 1) { // omni.unit:41, g_structa.act:712
		for _, st := range glob.Dats.ApModel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_hardware" && len(va) > 1) { // omni.unit:79, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Kernel_hardware" && len(va) > 1) { // omni.unit:121, g_structa.act:712
		for _, st := range glob.Dats.ApKernel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Constraint_target_hw" && len(va) > 1) { // omni.unit:162, g_structa.act:712
		for _, st := range glob.Dats.ApConstraint {
			if (st.Ktarget_hwp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Metric_target_hw" && len(va) > 1) { // omni.unit:172, g_structa.act:712
		for _, st := range glob.Dats.ApMetric {
			if (st.Ktarget_hwp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Fusion_hardware" && len(va) > 1) { // omni.unit:188, g_structa.act:712
		for _, st := range glob.Dats.ApFusion {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:26, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Hardware > omni.unit:26, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Hardware > omni.unit:26, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent_hw" {
		if me.Kparent_hwp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparent_hwp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "emulation" {
		if me.Kemulationp >= 0 {
			st := glob.Dats.ApHardware[ me.Kemulationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "noise_model" {
		if me.Knoise_modelp >= 0 {
			st := glob.Dats.ApConstraint[ me.Knoise_modelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Project_hardware") { // omni.unit:16, g_structa.act:597
		for _, st := range glob.Dats.ApProject {
			if (st.Khardwarep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Hardware_parent_hw") { // omni.unit:31, g_structa.act:597
		for _, st := range glob.Dats.ApHardware {
			if (st.Kparent_hwp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Hardware_emulation") { // omni.unit:32, g_structa.act:597
		for _, st := range glob.Dats.ApHardware {
			if (st.Kemulationp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Model_hardware") { // omni.unit:41, g_structa.act:597
		for _, st := range glob.Dats.ApModel {
			if (st.Khardwarep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Op_hardware") { // omni.unit:79, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Khardwarep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Kernel_hardware") { // omni.unit:121, g_structa.act:597
		for _, st := range glob.Dats.ApKernel {
			if (st.Khardwarep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Constraint_target_hw") { // omni.unit:162, g_structa.act:597
		for _, st := range glob.Dats.ApConstraint {
			if (st.Ktarget_hwp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Metric_target_hw") { // omni.unit:172, g_structa.act:597
		for _, st := range glob.Dats.ApMetric {
			if (st.Ktarget_hwp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Fusion_hardware") { // omni.unit:188, g_structa.act:597
		for _, st := range glob.Dats.ApFusion {
			if (st.Khardwarep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Hardware %s,%s > omni.unit:26, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpModel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Khardwarep int
	Ksearch_spacep int
	Kconfigp int
	ItsLayer [] *KpLayer 
	ItsTensor [] *KpTensor 
	ItsOp [] *KpOp 
	Childs [] Kp
}

func (me KpModel) TypeName() string {
    return me.Comp
}
func (me KpModel) GetLineNo() string {
	return me.LineNo
}

func loadModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpModel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApModel)
	st.LineNo = lno
	st.Comp = "Model";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	st.Ksearch_spacep = -1
	st.Kconfigp = -1
	name,_ := st.Names["model"].(string)
	st.Names["_key"] = "model"
	act.index["Model_" + name] = st.Me;
	st.MyName = name
	act.ApModel = append(act.ApModel, st)
	return 0
}

func (me KpModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // omni.unit:41, g_structa.act:623
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "search_space" { // omni.unit:42, g_structa.act:623
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "config" { // omni.unit:43, g_structa.act:623
		if (me.Kconfigp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kconfigp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Project_model" && len(va) > 1) { // omni.unit:14, g_structa.act:712
		for _, st := range glob.Dats.ApProject {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Block_model" && len(va) > 1) { // omni.unit:60, g_structa.act:712
		for _, st := range glob.Dats.ApBlock {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Arg_model" && len(va) > 1) { // omni.unit:104, g_structa.act:712
		for _, st := range glob.Dats.ApArg {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchSpace_target_model" && len(va) > 1) { // omni.unit:137, g_structa.act:712
		for _, st := range glob.Dats.ApSearchSpace {
			if (st.Ktarget_modelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Constraint_model" && len(va) > 1) { // omni.unit:163, g_structa.act:712
		for _, st := range glob.Dats.ApConstraint {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Checkpoint_model" && len(va) > 1) { // omni.unit:180, g_structa.act:712
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ControlFlow_model1" && len(va) > 1) { // omni.unit:196, g_structa.act:712
		for _, st := range glob.Dats.ApControlFlow {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ControlFlow_model2" && len(va) > 1) { // omni.unit:198, g_structa.act:712
		for _, st := range glob.Dats.ApControlFlow {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ControlFlow_model3" && len(va) > 1) { // omni.unit:200, g_structa.act:712
		for _, st := range glob.Dats.ApControlFlow {
			if (st.Kmodel3p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:36, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Model > omni.unit:36, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Model > omni.unit:36, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpModel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Layer" { // omni.unit:47, g_structa.act:690
		for _, st := range me.ItsLayer {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if (ret != 0) {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Tensor" { // omni.unit:63, g_structa.act:690
		for _, st := range me.ItsTensor {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if (ret != 0) {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "Op" { // omni.unit:72, g_structa.act:690
		for _, st := range me.ItsOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if (ret != 0) {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "hardware" {
		if me.Khardwarep >= 0 {
			st := glob.Dats.ApHardware[ me.Khardwarep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "search_space" {
		if me.Ksearch_spacep >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Ksearch_spacep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "config" {
		if me.Kconfigp >= 0 {
			st := glob.Dats.ApConfig[ me.Kconfigp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Project_model") { // omni.unit:14, g_structa.act:597
		for _, st := range glob.Dats.ApProject {
			if (st.Kmodelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Block_model") { // omni.unit:60, g_structa.act:597
		for _, st := range glob.Dats.ApBlock {
			if (st.Kmodelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Arg_model") { // omni.unit:104, g_structa.act:597
		for _, st := range glob.Dats.ApArg {
			if (st.Kmodelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "SearchSpace_target_model") { // omni.unit:137, g_structa.act:597
		for _, st := range glob.Dats.ApSearchSpace {
			if (st.Ktarget_modelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Constraint_model") { // omni.unit:163, g_structa.act:597
		for _, st := range glob.Dats.ApConstraint {
			if (st.Kmodelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Checkpoint_model") { // omni.unit:180, g_structa.act:597
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kmodelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "ControlFlow_model1") { // omni.unit:196, g_structa.act:597
		for _, st := range glob.Dats.ApControlFlow {
			if (st.Kmodel1p == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "ControlFlow_model2") { // omni.unit:198, g_structa.act:597
		for _, st := range glob.Dats.ApControlFlow {
			if (st.Kmodel2p == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "ControlFlow_model3") { // omni.unit:200, g_structa.act:597
		for _, st := range glob.Dats.ApControlFlow {
			if (st.Kmodel3p == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Model %s,%s > omni.unit:36, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpLayer struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kparent_layerp int
}

func (me KpLayer) TypeName() string {
    return me.Comp
}
func (me KpLayer) GetLineNo() string {
	return me.LineNo
}

func loadLayer(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpLayer)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApLayer)
	st.LineNo = lno
	st.Comp = "Layer";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparent_layerp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Layer has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Layer under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsLayer = append(act.ApModel[ len( act.ApModel )-1 ].ItsLayer, st)	// omni.unit:36, g_structa.act:416
	name,_ := st.Names["layer"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Layer_" + name	// omni.unit:51, g_structa.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApLayer = append(act.ApLayer, st)
	return 0
}

func (me KpLayer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "parent_layer" { // omni.unit:52, g_structa.act:623
		if (me.Kparent_layerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kparent_layerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // omni.unit:36, g_structa.act:586
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "Layer_parent_layer" && len(va) > 1) { // omni.unit:52, g_structa.act:712
		for _, st := range glob.Dats.ApLayer {
			if (st.Kparent_layerp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_layer" && len(va) > 1) { // omni.unit:87, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Klayerp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:47, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Layer > omni.unit:47, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Layer > omni.unit:47, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpLayer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // omni.unit:36, g_structa.act:571
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "parent_layer" {
		if me.Kparent_layerp >= 0 {
			st := glob.Dats.ApLayer[ me.Kparent_layerp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Layer_parent_layer") { // omni.unit:52, g_structa.act:597
		for _, st := range glob.Dats.ApLayer {
			if (st.Kparent_layerp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Op_layer") { // omni.unit:87, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Klayerp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Layer %s,%s > omni.unit:47, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBlock struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kmodelp int
}

func (me KpBlock) TypeName() string {
    return me.Comp
}
func (me KpBlock) GetLineNo() string {
	return me.LineNo
}

func loadBlock(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpBlock)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBlock)
	st.LineNo = lno
	st.Comp = "Block";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	name,_ := st.Names["block"].(string)
	st.Names["_key"] = "block"
	act.index["Block_" + name] = st.Me;
	st.MyName = name
	act.ApBlock = append(act.ApBlock, st)
	return 0
}

func (me KpBlock) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // omni.unit:60, g_structa.act:623
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // omni.unit:55, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBlock[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Block > omni.unit:55, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Block > omni.unit:55, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpBlock) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "model" {
		if me.Kmodelp >= 0 {
			st := glob.Dats.ApModel[ me.Kmodelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Block %s,%s > omni.unit:55, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTensor struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kproducerp int
	Kdistributionp int
}

func (me KpTensor) TypeName() string {
    return me.Comp
}
func (me KpTensor) GetLineNo() string {
	return me.LineNo
}

func loadTensor(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpTensor)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTensor)
	st.LineNo = lno
	st.Comp = "Tensor";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kproducerp = -1
	st.Kdistributionp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Tensor has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Tensor under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsTensor = append(act.ApModel[ len( act.ApModel )-1 ].ItsTensor, st)	// omni.unit:36, g_structa.act:416
	name,_ := st.Names["tensor"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Tensor_" + name	// omni.unit:67, g_structa.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTensor = append(act.ApTensor, st)
	return 0
}

func (me KpTensor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "producer" { // omni.unit:68, g_structa.act:623
		if (me.Kproducerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kproducerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "distribution" { // omni.unit:69, g_structa.act:623
		if (me.Kdistributionp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyFunction[ me.Kdistributionp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // omni.unit:36, g_structa.act:586
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "Op_predicate" && len(va) > 1) { // omni.unit:85, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Kpredicatep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:63, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Tensor > omni.unit:63, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Tensor > omni.unit:63, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpTensor) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // omni.unit:36, g_structa.act:571
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "producer" {
		if me.Kproducerp >= 0 {
			st := glob.Dats.ApOp[ me.Kproducerp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "distribution" {
		if me.Kdistributionp >= 0 {
			st := glob.Dats.ApEnergyFunction[ me.Kdistributionp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Op_predicate") { // omni.unit:85, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Kpredicatep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Arg_tensor") { // omni.unit:105, g_structa.act:597
		for _, st := range glob.Dats.ApArg {
			if (st.Ktensorp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Tensor %s,%s > omni.unit:63, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Khardwarep int
	Kenergy_fnp int
	Ksearch_spacep int
	Kstrategyp int
	Kpredicatep int
	Knext_opp int
	Klayerp int
	ItsArg [] *KpArg 
	ItsControlFlow [] *KpControlFlow 
	Childs [] Kp
}

func (me KpOp) TypeName() string {
    return me.Comp
}
func (me KpOp) GetLineNo() string {
	return me.LineNo
}

func loadOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOp)
	st.LineNo = lno
	st.Comp = "Op";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	st.Kenergy_fnp = -1
	st.Ksearch_spacep = -1
	st.Kstrategyp = -1
	st.Kpredicatep = -1
	st.Knext_opp = -1
	st.Klayerp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Op has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Op under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsOp = append(act.ApModel[ len( act.ApModel )-1 ].ItsOp, st)	// omni.unit:36, g_structa.act:416
	name,_ := st.Names["op"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Op_" + name	// omni.unit:76, g_structa.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOp = append(act.ApOp, st)
	return 0
}

func (me KpOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // omni.unit:79, g_structa.act:623
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "energy_fn" { // omni.unit:80, g_structa.act:623
		if (me.Kenergy_fnp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyFunction[ me.Kenergy_fnp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "search_space" { // omni.unit:81, g_structa.act:623
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "strategy" { // omni.unit:82, g_structa.act:623
		if (me.Kstrategyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApStrategy[ me.Kstrategyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "predicate" { // omni.unit:85, g_structa.act:623
		if (me.Kpredicatep >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kpredicatep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "next_op" { // omni.unit:86, g_structa.act:623
		if (me.Knext_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Knext_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "layer" { // omni.unit:87, g_structa.act:623
		if (me.Klayerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // omni.unit:36, g_structa.act:586
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "Tensor_producer" && len(va) > 1) { // omni.unit:68, g_structa.act:712
		for _, st := range glob.Dats.ApTensor {
			if (st.Kproducerp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_next_op" && len(va) > 1) { // omni.unit:86, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Knext_opp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:72, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Op > omni.unit:72, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Op > omni.unit:72, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Arg" { // omni.unit:98, g_structa.act:690
		for _, st := range me.ItsArg {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if (ret != 0) {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "ControlFlow" { // omni.unit:191, g_structa.act:690
		for _, st := range me.ItsControlFlow {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if (ret != 0) {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "parent" { // omni.unit:36, g_structa.act:571
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "hardware" {
		if me.Khardwarep >= 0 {
			st := glob.Dats.ApHardware[ me.Khardwarep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "energy_fn" {
		if me.Kenergy_fnp >= 0 {
			st := glob.Dats.ApEnergyFunction[ me.Kenergy_fnp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "search_space" {
		if me.Ksearch_spacep >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Ksearch_spacep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "strategy" {
		if me.Kstrategyp >= 0 {
			st := glob.Dats.ApStrategy[ me.Kstrategyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "predicate" {
		if me.Kpredicatep >= 0 {
			st := glob.Dats.ApTensor[ me.Kpredicatep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "next_op" {
		if me.Knext_opp >= 0 {
			st := glob.Dats.ApOp[ me.Knext_opp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "layer" {
		if me.Klayerp >= 0 {
			st := glob.Dats.ApLayer[ me.Klayerp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Tensor_producer") { // omni.unit:68, g_structa.act:597
		for _, st := range glob.Dats.ApTensor {
			if (st.Kproducerp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Op_next_op") { // omni.unit:86, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Knext_opp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Constraint_target_op") { // omni.unit:164, g_structa.act:597
		for _, st := range glob.Dats.ApConstraint {
			if (st.Ktarget_opp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Op %s,%s > omni.unit:72, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpArg struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kmodelp int
	Ktensorp int
}

func (me KpArg) TypeName() string {
    return me.Comp
}
func (me KpArg) GetLineNo() string {
	return me.LineNo
}

func loadArg(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpArg)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApArg)
	st.LineNo = lno
	st.Comp = "Arg";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Ktensorp = -1
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Arg has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Arg under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsArg = append(act.ApOp[ len( act.ApOp )-1 ].ItsArg, st)	// omni.unit:72, g_structa.act:416
	name,_ := st.Names["arg"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Arg_" + name	// omni.unit:102, g_structa.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArg = append(act.ApArg, st)
	return 0
}

func (me KpArg) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // omni.unit:104, g_structa.act:623
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "tensor" { // omni.unit:105, g_structa.act:623
		if (me.Ktensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // omni.unit:72, g_structa.act:586
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // omni.unit:98, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArg[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Arg > omni.unit:98, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Arg > omni.unit:98, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpArg) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // omni.unit:72, g_structa.act:571
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "model" {
		if me.Kmodelp >= 0 {
			st := glob.Dats.ApModel[ me.Kmodelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "tensor" {
		if me.Ktensorp >= 0 {
			st := glob.Dats.ApTensor[ me.Ktensorp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Arg %s,%s > omni.unit:98, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpConfig struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kschedulep int
}

func (me KpConfig) TypeName() string {
    return me.Comp
}
func (me KpConfig) GetLineNo() string {
	return me.LineNo
}

func loadConfig(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpConfig)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApConfig)
	st.LineNo = lno
	st.Comp = "Config";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kschedulep = -1
	name,_ := st.Names["config"].(string)
	st.Names["_key"] = "config"
	act.index["Config_" + name] = st.Me;
	st.MyName = name
	act.ApConfig = append(act.ApConfig, st)
	return 0
}

func (me KpConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "schedule" { // omni.unit:112, g_structa.act:623
		if (me.Kschedulep >= 0 && len(va) > 1) {
			return( glob.Dats.ApStrategy[ me.Kschedulep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Model_config" && len(va) > 1) { // omni.unit:43, g_structa.act:712
		for _, st := range glob.Dats.ApModel {
			if (st.Kconfigp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:107, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Config > omni.unit:107, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Config > omni.unit:107, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "schedule" {
		if me.Kschedulep >= 0 {
			st := glob.Dats.ApStrategy[ me.Kschedulep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Model_config") { // omni.unit:43, g_structa.act:597
		for _, st := range glob.Dats.ApModel {
			if (st.Kconfigp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Config %s,%s > omni.unit:107, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpKernel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Khardwarep int
}

func (me KpKernel) TypeName() string {
    return me.Comp
}
func (me KpKernel) GetLineNo() string {
	return me.LineNo
}

func loadKernel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpKernel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApKernel)
	st.LineNo = lno
	st.Comp = "Kernel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	name,_ := st.Names["kernel"].(string)
	st.Names["_key"] = "kernel"
	act.index["Kernel_" + name] = st.Me;
	st.MyName = name
	act.ApKernel = append(act.ApKernel, st)
	return 0
}

func (me KpKernel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // omni.unit:121, g_structa.act:623
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // omni.unit:115, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Kernel > omni.unit:115, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Kernel > omni.unit:115, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "hardware" {
		if me.Khardwarep >= 0 {
			st := glob.Dats.ApHardware[ me.Khardwarep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Kernel %s,%s > omni.unit:115, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnergyFunction struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
}

func (me KpEnergyFunction) TypeName() string {
    return me.Comp
}
func (me KpEnergyFunction) GetLineNo() string {
	return me.LineNo
}

func loadEnergyFunction(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpEnergyFunction)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnergyFunction)
	st.LineNo = lno
	st.Comp = "EnergyFunction";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["energy_fn"].(string)
	st.Names["_key"] = "energy_fn"
	act.index["EnergyFunction_" + name] = st.Me;
	st.MyName = name
	act.ApEnergyFunction = append(act.ApEnergyFunction, st)
	return 0
}

func (me KpEnergyFunction) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Tensor_distribution" && len(va) > 1) { // omni.unit:69, g_structa.act:712
		for _, st := range glob.Dats.ApTensor {
			if (st.Kdistributionp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_energy_fn" && len(va) > 1) { // omni.unit:80, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Kenergy_fnp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:124, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyFunction[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnergyFunction > omni.unit:124, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,EnergyFunction > omni.unit:124, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpEnergyFunction) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Tensor_distribution") { // omni.unit:69, g_structa.act:597
		for _, st := range glob.Dats.ApTensor {
			if (st.Kdistributionp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Op_energy_fn") { // omni.unit:80, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Kenergy_fnp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for EnergyFunction %s,%s > omni.unit:124, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchSpace struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Ktarget_modelp int
	ItsDimension [] *KpDimension 
	Childs [] Kp
}

func (me KpSearchSpace) TypeName() string {
    return me.Comp
}
func (me KpSearchSpace) GetLineNo() string {
	return me.LineNo
}

func loadSearchSpace(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpSearchSpace)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchSpace)
	st.LineNo = lno
	st.Comp = "SearchSpace";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ktarget_modelp = -1
	name,_ := st.Names["space"].(string)
	st.Names["_key"] = "space"
	act.index["SearchSpace_" + name] = st.Me;
	st.MyName = name
	act.ApSearchSpace = append(act.ApSearchSpace, st)
	return 0
}

func (me KpSearchSpace) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target_model" { // omni.unit:137, g_structa.act:623
		if (me.Ktarget_modelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Ktarget_modelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Model_search_space" && len(va) > 1) { // omni.unit:42, g_structa.act:712
		for _, st := range glob.Dats.ApModel {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_search_space" && len(va) > 1) { // omni.unit:81, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Strategy_search_space" && len(va) > 1) { // omni.unit:152, g_structa.act:712
		for _, st := range glob.Dats.ApStrategy {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:132, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpace > omni.unit:132, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,SearchSpace > omni.unit:132, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpSearchSpace) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Dimension" { // omni.unit:140, g_structa.act:690
		for _, st := range me.ItsDimension {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
					return(ret)
				}
				continue
			}
			ret := GoAct(glob, st)
			if (ret != 0) {
				return(ret)
			}
		}
		return(0)
	}
	if va[0] == "target_model" {
		if me.Ktarget_modelp >= 0 {
			st := glob.Dats.ApModel[ me.Ktarget_modelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Model_search_space") { // omni.unit:42, g_structa.act:597
		for _, st := range glob.Dats.ApModel {
			if (st.Ksearch_spacep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Op_search_space") { // omni.unit:81, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Ksearch_spacep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Strategy_search_space") { // omni.unit:152, g_structa.act:597
		for _, st := range glob.Dats.ApStrategy {
			if (st.Ksearch_spacep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchSpace %s,%s > omni.unit:132, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDimension struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
}

func (me KpDimension) TypeName() string {
    return me.Comp
}
func (me KpDimension) GetLineNo() string {
	return me.LineNo
}

func loadDimension(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpDimension)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDimension)
	st.LineNo = lno
	st.Comp = "Dimension";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Dimension has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Dimension under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsDimension = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsDimension, st)	// omni.unit:132, g_structa.act:416
	name,_ := st.Names["dimension"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Dimension_" + name	// omni.unit:144, g_structa.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDimension = append(act.ApDimension, st)
	return 0
}

func (me KpDimension) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // omni.unit:132, g_structa.act:586
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // omni.unit:140, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDimension[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Dimension > omni.unit:140, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Dimension > omni.unit:140, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpDimension) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // omni.unit:132, g_structa.act:571
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Dimension %s,%s > omni.unit:140, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpStrategy struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Ksearch_spacep int
	Kfitnessp int
}

func (me KpStrategy) TypeName() string {
    return me.Comp
}
func (me KpStrategy) GetLineNo() string {
	return me.LineNo
}

func loadStrategy(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpStrategy)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApStrategy)
	st.LineNo = lno
	st.Comp = "Strategy";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ksearch_spacep = -1
	st.Kfitnessp = -1
	name,_ := st.Names["strategy"].(string)
	st.Names["_key"] = "strategy"
	act.index["Strategy_" + name] = st.Me;
	st.MyName = name
	act.ApStrategy = append(act.ApStrategy, st)
	return 0
}

func (me KpStrategy) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "search_space" { // omni.unit:152, g_structa.act:623
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "fitness" { // omni.unit:153, g_structa.act:623
		if (me.Kfitnessp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMetric[ me.Kfitnessp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Project_strategy" && len(va) > 1) { // omni.unit:15, g_structa.act:712
		for _, st := range glob.Dats.ApProject {
			if (st.Kstrategyp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_strategy" && len(va) > 1) { // omni.unit:82, g_structa.act:712
		for _, st := range glob.Dats.ApOp {
			if (st.Kstrategyp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Config_schedule" && len(va) > 1) { // omni.unit:112, g_structa.act:712
		for _, st := range glob.Dats.ApConfig {
			if (st.Kschedulep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:147, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApStrategy[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Strategy > omni.unit:147, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Strategy > omni.unit:147, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpStrategy) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "search_space" {
		if me.Ksearch_spacep >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Ksearch_spacep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "fitness" {
		if me.Kfitnessp >= 0 {
			st := glob.Dats.ApMetric[ me.Kfitnessp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Project_strategy") { // omni.unit:15, g_structa.act:597
		for _, st := range glob.Dats.ApProject {
			if (st.Kstrategyp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Op_strategy") { // omni.unit:82, g_structa.act:597
		for _, st := range glob.Dats.ApOp {
			if (st.Kstrategyp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Config_schedule") { // omni.unit:112, g_structa.act:597
		for _, st := range glob.Dats.ApConfig {
			if (st.Kschedulep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Strategy %s,%s > omni.unit:147, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpConstraint struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Ktarget_hwp int
	Kmodelp int
	Ktarget_opp int
}

func (me KpConstraint) TypeName() string {
    return me.Comp
}
func (me KpConstraint) GetLineNo() string {
	return me.LineNo
}

func loadConstraint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpConstraint)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApConstraint)
	st.LineNo = lno
	st.Comp = "Constraint";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ktarget_hwp = -1
	st.Kmodelp = -1
	st.Ktarget_opp = -1
	name,_ := st.Names["constraint_id"].(string)
	st.Names["_key"] = "constraint_id"
	act.index["Constraint_" + name] = st.Me;
	st.MyName = name
	act.ApConstraint = append(act.ApConstraint, st)
	return 0
}

func (me KpConstraint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target_hw" { // omni.unit:162, g_structa.act:623
		if (me.Ktarget_hwp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Ktarget_hwp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model" { // omni.unit:163, g_structa.act:623
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "target_op" { // omni.unit:164, g_structa.act:623
		if (me.Ktarget_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Ktarget_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Hardware_noise_model" && len(va) > 1) { // omni.unit:33, g_structa.act:712
		for _, st := range glob.Dats.ApHardware {
			if (st.Knoise_modelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:157, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApConstraint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Constraint > omni.unit:157, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Constraint > omni.unit:157, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpConstraint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "target_hw" {
		if me.Ktarget_hwp >= 0 {
			st := glob.Dats.ApHardware[ me.Ktarget_hwp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "model" {
		if me.Kmodelp >= 0 {
			st := glob.Dats.ApModel[ me.Kmodelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "target_op" {
		if me.Ktarget_opp >= 0 {
			st := glob.Dats.ApOp[ me.Ktarget_opp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Hardware_noise_model") { // omni.unit:33, g_structa.act:597
		for _, st := range glob.Dats.ApHardware {
			if (st.Knoise_modelp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Constraint %s,%s > omni.unit:157, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMetric struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Ktarget_hwp int
}

func (me KpMetric) TypeName() string {
    return me.Comp
}
func (me KpMetric) GetLineNo() string {
	return me.LineNo
}

func loadMetric(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpMetric)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMetric)
	st.LineNo = lno
	st.Comp = "Metric";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ktarget_hwp = -1
	name,_ := st.Names["metric"].(string)
	st.Names["_key"] = "metric"
	act.index["Metric_" + name] = st.Me;
	st.MyName = name
	act.ApMetric = append(act.ApMetric, st)
	return 0
}

func (me KpMetric) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target_hw" { // omni.unit:172, g_structa.act:623
		if (me.Ktarget_hwp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Ktarget_hwp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Strategy_fitness" && len(va) > 1) { // omni.unit:153, g_structa.act:712
		for _, st := range glob.Dats.ApStrategy {
			if (st.Kfitnessp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // omni.unit:167, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMetric[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Metric > omni.unit:167, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Metric > omni.unit:167, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpMetric) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "target_hw" {
		if me.Ktarget_hwp >= 0 {
			st := glob.Dats.ApHardware[ me.Ktarget_hwp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Strategy_fitness") { // omni.unit:153, g_structa.act:597
		for _, st := range glob.Dats.ApStrategy {
			if (st.Kfitnessp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Metric %s,%s > omni.unit:167, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCheckpoint struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kmodelp int
}

func (me KpCheckpoint) TypeName() string {
    return me.Comp
}
func (me KpCheckpoint) GetLineNo() string {
	return me.LineNo
}

func loadCheckpoint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpCheckpoint)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCheckpoint)
	st.LineNo = lno
	st.Comp = "Checkpoint";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	name,_ := st.Names["checkpoint_id"].(string)
	st.Names["_key"] = "checkpoint_id"
	act.index["Checkpoint_" + name] = st.Me;
	st.MyName = name
	act.ApCheckpoint = append(act.ApCheckpoint, st)
	return 0
}

func (me KpCheckpoint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // omni.unit:180, g_structa.act:623
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // omni.unit:175, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCheckpoint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Checkpoint > omni.unit:175, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Checkpoint > omni.unit:175, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpCheckpoint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "model" {
		if me.Kmodelp >= 0 {
			st := glob.Dats.ApModel[ me.Kmodelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Checkpoint %s,%s > omni.unit:175, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFusion struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Khardwarep int
}

func (me KpFusion) TypeName() string {
    return me.Comp
}
func (me KpFusion) GetLineNo() string {
	return me.LineNo
}

func loadFusion(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpFusion)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFusion)
	st.LineNo = lno
	st.Comp = "Fusion";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	name,_ := st.Names["fusion"].(string)
	st.Names["_key"] = "fusion"
	act.index["Fusion_" + name] = st.Me;
	st.MyName = name
	act.ApFusion = append(act.ApFusion, st)
	return 0
}

func (me KpFusion) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // omni.unit:188, g_structa.act:623
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // omni.unit:183, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusion[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Fusion > omni.unit:183, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,Fusion > omni.unit:183, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpFusion) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "hardware" {
		if me.Khardwarep >= 0 {
			st := glob.Dats.ApHardware[ me.Khardwarep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Fusion %s,%s > omni.unit:183, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpControlFlow struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kmodel1p int
	Kmodel2p int
	Kmodel3p int
}

func (me KpControlFlow) TypeName() string {
    return me.Comp
}
func (me KpControlFlow) GetLineNo() string {
	return me.LineNo
}

func loadControlFlow(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpControlFlow)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApControlFlow)
	st.LineNo = lno
	st.Comp = "ControlFlow";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kmodel2p = -1
	st.Kmodel3p = -1
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ControlFlow has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " ControlFlow under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsControlFlow = append(act.ApOp[ len( act.ApOp )-1 ].ItsControlFlow, st)	// omni.unit:72, g_structa.act:416
	name,_ := st.Names["control"].(string)
	s := strconv.Itoa(st.Kparentp) + "_ControlFlow_" + name	// omni.unit:195, g_structa.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApControlFlow = append(act.ApControlFlow, st)
	return 0
}

func (me KpControlFlow) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // omni.unit:196, g_structa.act:623
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // omni.unit:198, g_structa.act:623
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model3" { // omni.unit:200, g_structa.act:623
		if (me.Kmodel3p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel3p ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // omni.unit:72, g_structa.act:586
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // omni.unit:191, g_structa.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ControlFlow > omni.unit:191, g_structa.act:185?", va[0], lno, me.LineNo)
		return false, msg
	}
	if va[0] == "payload" {
		tmp := maps.Clone(me.Names)
		delete(tmp, "kMe")
		delete(tmp, "kComp")
		jsonData, _ := json.MarshalIndent(tmp, "   ", "  ")
		return true, string(jsonData)
	}
	r := me.Names[va[0]]
	if r == nil { 
		rr := fmt.Sprintf("?%s?:%s,%s,ControlFlow > omni.unit:191, g_structa.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpControlFlow) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // omni.unit:72, g_structa.act:571
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "model1" {
		if me.Kmodel1p >= 0 {
			st := glob.Dats.ApModel[ me.Kmodel1p ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "model2" {
		if me.Kmodel2p >= 0 {
			st := glob.Dats.ApModel[ me.Kmodel2p ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "model3" {
		if me.Kmodel3p >= 0 {
			st := glob.Dats.ApModel[ me.Kmodel3p ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ControlFlow %s,%s > omni.unit:191, g_structa.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpActor struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kname string
	Kcomp string
	Kattr string
	Keq string
	Kvalue string
	Childs [] Kp
}

func loadActor(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpActor)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApActor)
	st.LineNo = lno
	st.Comp = "Actor";
	st.Flags = flag;
	p,st.Kname = getw(ln,p)
	p,st.Kcomp = getw(ln,p)
	p,st.Kattr = getw(ln,p)
	p,st.Keq = getw(ln,p)
	p,st.Kvalue = getws(ln,p)
	act.index["Actor_" + st.Kname] = st.Me;
	act.ApActor = append(act.ApActor, st)
	return 0
}

type KpAll struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kwhat string
	Kactor string
	Kargs string
	Kactorp int
}

func loadAll(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpAll)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAll)
	st.LineNo = lno
	st.Comp = "All";
	st.Flags = flag;
	p,st.Kwhat = getw(ln,p)
	p,st.Kactor = getw(ln,p)
	p,st.Kargs = getws(ln,p)
	st.Kactorp = -1
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " All has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApAll = append(act.ApAll, st)
	return 0
}

type KpDu struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kactor string
	Kargs string
	Kactorp int
}

func loadDu(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpDu)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDu)
	st.LineNo = lno
	st.Comp = "Du";
	st.Flags = flag;
	p,st.Kactor = getw(ln,p)
	p,st.Kargs = getws(ln,p)
	st.Kactorp = -1
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Du has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApDu = append(act.ApDu, st)
	return 0
}

type KpNew struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kwhere string
	Kwhat string
	Kline string
}

func loadNew(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpNew)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNew)
	st.LineNo = lno
	st.Comp = "New";
	st.Flags = flag;
	p,st.Kwhere = getw(ln,p)
	p,st.Kwhat = getw(ln,p)
	p,st.Kline = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " New has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApNew = append(act.ApNew, st)
	return 0
}

type KpRefs struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kwhere string
}

func loadRefs(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpRefs)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApRefs)
	st.LineNo = lno
	st.Comp = "Refs";
	st.Flags = flag;
	p,st.Kwhere = getw(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Refs has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApRefs = append(act.ApRefs, st)
	return 0
}

type KpVar struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kattr string
	Keq string
	Kvalue string
}

func loadVar(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpVar)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApVar)
	st.LineNo = lno
	st.Comp = "Var";
	st.Flags = flag;
	p,st.Kattr = getw(ln,p)
	p,st.Keq = getw(ln,p)
	p,st.Kvalue = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Var has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApVar = append(act.ApVar, st)
	return 0
}

type KpIts struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kwhat string
	Kactor string
	Kargs string
	Kactorp int
}

func loadIts(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpIts)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApIts)
	st.LineNo = lno
	st.Comp = "Its";
	st.Flags = flag;
	p,st.Kwhat = getw(ln,p)
	p,st.Kactor = getw(ln,p)
	p,st.Kargs = getws(ln,p)
	st.Kactorp = -1
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Its has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApIts = append(act.ApIts, st)
	return 0
}

type KpC struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kdesc string
}

func loadC(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpC)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApC)
	st.LineNo = lno
	st.Comp = "C";
	st.Flags = flag;
	p,st.Kdesc = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " C has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApC = append(act.ApC, st)
	return 0
}

type KpCs struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kdesc string
}

func loadCs(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpCs)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCs)
	st.LineNo = lno
	st.Comp = "Cs";
	st.Flags = flag;
	p,st.Kdesc = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Cs has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApCs = append(act.ApCs, st)
	return 0
}

type KpOut struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kwhat string
	Kpad string
	Kdesc string
}

func loadOut(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpOut)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOut)
	st.LineNo = lno
	st.Comp = "Out";
	st.Flags = flag;
	p,st.Kwhat = getw(ln,p)
	p,st.Kpad = getw(ln,p)
	p,st.Kdesc = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Out has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApOut = append(act.ApOut, st)
	return 0
}

type KpIn struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kflag string
}

func loadIn(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpIn)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApIn)
	st.LineNo = lno
	st.Comp = "In";
	st.Flags = flag;
	p,st.Kflag = getw(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " In has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApIn = append(act.ApIn, st)
	return 0
}

type KpBreak struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kwhat string
	Kpad string
	Kactor string
	Kcheck string
}

func loadBreak(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpBreak)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBreak)
	st.LineNo = lno
	st.Comp = "Break";
	st.Flags = flag;
	p,st.Kwhat = getw(ln,p)
	p,st.Kpad = getw(ln,p)
	p,st.Kactor = getw(ln,p)
	p,st.Kcheck = getw(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Break has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApBreak = append(act.ApBreak, st)
	return 0
}

type KpAdd struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kpath string
	Kdata string
}

func loadAdd(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpAdd)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAdd)
	st.LineNo = lno
	st.Comp = "Add";
	st.Flags = flag;
	p,st.Kpath = getw(ln,p)
	p,st.Kdata = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Add has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApAdd = append(act.ApAdd, st)
	return 0
}

type KpThis struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kpath string
	Kactor string
	Kargs string
	Kactorp int
}

func loadThis(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpThis)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApThis)
	st.LineNo = lno
	st.Comp = "This";
	st.Flags = flag;
	p,st.Kpath = getw(ln,p)
	p,st.Kactor = getw(ln,p)
	p,st.Kargs = getws(ln,p)
	st.Kactorp = -1
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " This has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApThis = append(act.ApThis, st)
	return 0
}

type KpReplace struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Kparentp int
	Kpath string
	Kpad string
	Kwith string
	Kpad2 string
	Kmatch string
}

func loadReplace(act *ActT, ln string, pos int, lno string, flag []string) int {
	p := pos
	st := new(KpReplace)
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApReplace)
	st.LineNo = lno
	st.Comp = "Replace";
	st.Flags = flag;
	p,st.Kpath = getw(ln,p)
	p,st.Kpad = getw(ln,p)
	p,st.Kwith = getw(ln,p)
	p,st.Kpad2 = getw(ln,p)
	p,st.Kmatch = getws(ln,p)
	st.Kparentp = len(act.ApActor)-1;
	if (st.Kparentp < 0 ) { 
		print(lno + " Replace has no Actor parent\n") ;
		return 1
	}
	act.ApActor[ len( act.ApActor )-1 ].Childs = append(act.ApActor[ len( act.ApActor )-1 ].Childs, st)
	act.ApReplace = append(act.ApReplace, st)
	return 0
}

