package main

import (
	"fmt"
	"strconv"
)

type Kp interface {
	DoIts(glob *GlobT, va []string, lno string) int
	GetVar(glob *GlobT, va []string, lno string) (bool, string)
	GetLineNo() string
	TypeName() string
}

type KpDomain struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDomain) TypeName() string {
    return me.Comp
}
func (me KpDomain) GetLineNo() string {
	return me.LineNo
}

func loadDomain(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["name"]
	act.index["Domain_" + name] = st.Me;
	st.MyName = name
	act.ApDomain = append(act.ApDomain, st)
	return 0
}

func (me KpDomain) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:5, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Domain > net5.unit:5, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Domain > net5.unit:5, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDomain) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for Domain %s,%s > net5.unit:5, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	ItsKernelParam [] *KpKernelParam 
	ItsKernelOp [] *KpKernelOp 
	Childs [] Kp
}

func (me KpKernel) TypeName() string {
    return me.Comp
}
func (me KpKernel) GetLineNo() string {
	return me.LineNo
}

func loadKernel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["kernel"]
	act.index["Kernel_" + name] = st.Me;
	st.MyName = name
	act.ApKernel = append(act.ApKernel, st)
	return 0
}

func (me KpKernel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Op_kernel" && len(va) > 1) { // net5.unit:190, g_structh.act:698
		for _, st := range glob.Dats.ApOp {
			if (st.Kkernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // net5.unit:14, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Kernel > net5.unit:14, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Kernel > net5.unit:14, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "KernelParam" { // net5.unit:27, g_structh.act:676
		for _, st := range me.ItsKernelParam {
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
	if va[0] == "KernelOp" { // net5.unit:37, g_structh.act:676
		for _, st := range me.ItsKernelOp {
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
	if (va[0] == "Op_kernel") { // net5.unit:190, g_structh.act:583
		for _, st := range glob.Dats.ApOp {
			if (st.Kkernelp == me.Me) {
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
	        fmt.Printf("?No its %s for Kernel %s,%s > net5.unit:14, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpKernelParam struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpKernelParam) TypeName() string {
    return me.Comp
}
func (me KpKernelParam) GetLineNo() string {
	return me.LineNo
}

func loadKernelParam(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpKernelParam)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApKernelParam)
	st.LineNo = lno
	st.Comp = "KernelParam";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApKernel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " KernelParam has no Kernel parent\n") ;
		return 1
	}
	st.Parent = act.ApKernel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " KernelParam under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApKernel[ len( act.ApKernel )-1 ].Childs = append(act.ApKernel[ len( act.ApKernel )-1 ].Childs, st)
	act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelParam = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelParam, st)	// net5.unit:14, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_KernelParam_" + name	// net5.unit:32, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApKernelParam = append(act.ApKernelParam, st)
	return 0
}

func (me KpKernelParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:14, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:27, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelParam > net5.unit:27, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelParam > net5.unit:27, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:14, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for KernelParam %s,%s > net5.unit:27, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpKernelOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpKernelOp) TypeName() string {
    return me.Comp
}
func (me KpKernelOp) GetLineNo() string {
	return me.LineNo
}

func loadKernelOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpKernelOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApKernelOp)
	st.LineNo = lno
	st.Comp = "KernelOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApKernel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " KernelOp has no Kernel parent\n") ;
		return 1
	}
	st.Parent = act.ApKernel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " KernelOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApKernel[ len( act.ApKernel )-1 ].Childs = append(act.ApKernel[ len( act.ApKernel )-1 ].Childs, st)
	act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelOp = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelOp, st)	// net5.unit:14, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_KernelOp_" + name	// net5.unit:43, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApKernelOp = append(act.ApKernelOp, st)
	return 0
}

func (me KpKernelOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:14, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:37, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelOp > net5.unit:37, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelOp > net5.unit:37, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:14, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Op_kernel_op") { // net5.unit:191, g_structh.act:583
		for _, st := range glob.Dats.ApOp {
			if (st.Kkernel_opp == me.Me) {
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
	        fmt.Printf("?No its %s for KernelOp %s,%s > net5.unit:37, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOptimization struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kflagsp int
}

func (me KpOptimization) TypeName() string {
    return me.Comp
}
func (me KpOptimization) GetLineNo() string {
	return me.LineNo
}

func loadOptimization(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOptimization)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOptimization)
	st.LineNo = lno
	st.Comp = "Optimization";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kflagsp = -1
	name,_ := st.Names["target"]
	act.index["Optimization_" + name] = st.Me;
	st.MyName = name
	act.ApOptimization = append(act.ApOptimization, st)
	return 0
}

func (me KpOptimization) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "flags" { // net5.unit:57, g_structh.act:609
		if (me.Kflagsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFlagRule[ me.Kflagsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // net5.unit:51, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimization[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Optimization > net5.unit:51, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Optimization > net5.unit:51, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOptimization) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "flags" {
		if me.Kflagsp >= 0 {
			st := glob.Dats.ApFlagRule[ me.Kflagsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Optimization %s,%s > net5.unit:51, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpScheduleOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpScheduleOp) TypeName() string {
    return me.Comp
}
func (me KpScheduleOp) GetLineNo() string {
	return me.LineNo
}

func loadScheduleOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpScheduleOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApScheduleOp)
	st.LineNo = lno
	st.Comp = "ScheduleOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["schedule_op"]
	act.index["ScheduleOp_" + name] = st.Me;
	st.MyName = name
	act.ApScheduleOp = append(act.ApScheduleOp, st)
	return 0
}

func (me KpScheduleOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:61, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApScheduleOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ScheduleOp > net5.unit:61, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ScheduleOp > net5.unit:61, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpScheduleOp) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ScheduleOp %s,%s > net5.unit:61, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMemLayout struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpMemLayout) TypeName() string {
    return me.Comp
}
func (me KpMemLayout) GetLineNo() string {
	return me.LineNo
}

func loadMemLayout(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMemLayout)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMemLayout)
	st.LineNo = lno
	st.Comp = "MemLayout";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["target"]
	act.index["MemLayout_" + name] = st.Me;
	st.MyName = name
	act.ApMemLayout = append(act.ApMemLayout, st)
	return 0
}

func (me KpMemLayout) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:71, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMemLayout[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MemLayout > net5.unit:71, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MemLayout > net5.unit:71, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMemLayout) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for MemLayout %s,%s > net5.unit:71, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
}

func (me KpFusion) TypeName() string {
    return me.Comp
}
func (me KpFusion) GetLineNo() string {
	return me.LineNo
}

func loadFusion(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["fusion"]
	act.index["Fusion_" + name] = st.Me;
	st.MyName = name
	act.ApFusion = append(act.ApFusion, st)
	return 0
}

func (me KpFusion) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:81, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusion[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Fusion > net5.unit:81, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Fusion > net5.unit:81, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFusion) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for Fusion %s,%s > net5.unit:81, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	ItsTensor [] *KpTensor 
	ItsLayer [] *KpLayer 
	ItsConfig [] *KpConfig 
	ItsControlFlow [] *KpControlFlow 
	ItsGraphTensor [] *KpGraphTensor 
	ItsContinuousLayer [] *KpContinuousLayer 
	ItsSearchSpace [] *KpSearchSpace 
	ItsArchitectureParam [] *KpArchitectureParam 
	ItsArchitectureGradient [] *KpArchitectureGradient 
	ItsHyperNetwork [] *KpHyperNetwork 
	ItsMemoryMatrix [] *KpMemoryMatrix 
	ItsDifferentiableProgram [] *KpDifferentiableProgram 
	ItsMetaLearner [] *KpMetaLearner 
	ItsSparseExpertSystem [] *KpSparseExpertSystem 
	ItsNeuralSDE [] *KpNeuralSDE 
	ItsNeuralPDE [] *KpNeuralPDE 
	ItsDynamicGraphNetwork [] *KpDynamicGraphNetwork 
	ItsNeuralProgram [] *KpNeuralProgram 
	ItsLiquidNetwork [] *KpLiquidNetwork 
	ItsSymbolicShape [] *KpSymbolicShape 
	ItsJITCompiler [] *KpJITCompiler 
	Childs [] Kp
}

func (me KpModel) TypeName() string {
    return me.Comp
}
func (me KpModel) GetLineNo() string {
	return me.LineNo
}

func loadModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["model"]
	act.index["Model_" + name] = st.Me;
	st.MyName = name
	act.ApModel = append(act.ApModel, st)
	return 0
}

func (me KpModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Block_model" && len(va) > 1) { // net5.unit:115, g_structh.act:698
		for _, st := range glob.Dats.ApBlock {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Arg_model" && len(va) > 1) { // net5.unit:202, g_structh.act:698
		for _, st := range glob.Dats.ApArg {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ExpertRoutingOp_model2" && len(va) > 1) { // net5.unit:375, g_structh.act:698
		for _, st := range glob.Dats.ApExpertRoutingOp {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Schedule_model1" && len(va) > 1) { // net5.unit:650, g_structh.act:698
		for _, st := range glob.Dats.ApSchedule {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Schedule_model2" && len(va) > 1) { // net5.unit:653, g_structh.act:698
		for _, st := range glob.Dats.ApSchedule {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Condition_model" && len(va) > 1) { // net5.unit:680, g_structh.act:698
		for _, st := range glob.Dats.ApCondition {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Branch_model" && len(va) > 1) { // net5.unit:692, g_structh.act:698
		for _, st := range glob.Dats.ApBranch {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "StateTransfer_model1" && len(va) > 1) { // net5.unit:722, g_structh.act:698
		for _, st := range glob.Dats.ApStateTransfer {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "StateTransfer_model2" && len(va) > 1) { // net5.unit:724, g_structh.act:698
		for _, st := range glob.Dats.ApStateTransfer {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ContinuousLayer_model" && len(va) > 1) { // net5.unit:743, g_structh.act:698
		for _, st := range glob.Dats.ApContinuousLayer {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchOp_model" && len(va) > 1) { // net5.unit:769, g_structh.act:698
		for _, st := range glob.Dats.ApSearchOp {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ArchitectureGradient_model" && len(va) > 1) { // net5.unit:804, g_structh.act:698
		for _, st := range glob.Dats.ApArchitectureGradient {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ArchitectureGradient_model2" && len(va) > 1) { // net5.unit:806, g_structh.act:698
		for _, st := range glob.Dats.ApArchitectureGradient {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "HyperNetwork_model1" && len(va) > 1) { // net5.unit:823, g_structh.act:698
		for _, st := range glob.Dats.ApHyperNetwork {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "HyperNetwork_model2" && len(va) > 1) { // net5.unit:827, g_structh.act:698
		for _, st := range glob.Dats.ApHyperNetwork {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "WeightGenerator_model1" && len(va) > 1) { // net5.unit:838, g_structh.act:698
		for _, st := range glob.Dats.ApWeightGenerator {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "WeightGenerator_model2" && len(va) > 1) { // net5.unit:840, g_structh.act:698
		for _, st := range glob.Dats.ApWeightGenerator {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ProgramInstruction_model1" && len(va) > 1) { // net5.unit:883, g_structh.act:698
		for _, st := range glob.Dats.ApProgramInstruction {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ProgramInstruction_model2" && len(va) > 1) { // net5.unit:885, g_structh.act:698
		for _, st := range glob.Dats.ApProgramInstruction {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "MetaLearner_model1" && len(va) > 1) { // net5.unit:897, g_structh.act:698
		for _, st := range glob.Dats.ApMetaLearner {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "MetaLearner_model2" && len(va) > 1) { // net5.unit:899, g_structh.act:698
		for _, st := range glob.Dats.ApMetaLearner {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SparseExpertSystem_model" && len(va) > 1) { // net5.unit:915, g_structh.act:698
		for _, st := range glob.Dats.ApSparseExpertSystem {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "CapacityAwareRouter_model1" && len(va) > 1) { // net5.unit:929, g_structh.act:698
		for _, st := range glob.Dats.ApCapacityAwareRouter {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "CapacityAwareRouter_model2" && len(va) > 1) { // net5.unit:931, g_structh.act:698
		for _, st := range glob.Dats.ApCapacityAwareRouter {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "CapacityAwareRouter_model3" && len(va) > 1) { // net5.unit:933, g_structh.act:698
		for _, st := range glob.Dats.ApCapacityAwareRouter {
			if (st.Kmodel3p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "NeuralSDE_model1" && len(va) > 1) { // net5.unit:948, g_structh.act:698
		for _, st := range glob.Dats.ApNeuralSDE {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "NeuralSDE_model2" && len(va) > 1) { // net5.unit:950, g_structh.act:698
		for _, st := range glob.Dats.ApNeuralSDE {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "NeuralPDE_model1" && len(va) > 1) { // net5.unit:963, g_structh.act:698
		for _, st := range glob.Dats.ApNeuralPDE {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "NeuralPDE_model2" && len(va) > 1) { // net5.unit:965, g_structh.act:698
		for _, st := range glob.Dats.ApNeuralPDE {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "NeuralPDE_model3" && len(va) > 1) { // net5.unit:967, g_structh.act:698
		for _, st := range glob.Dats.ApNeuralPDE {
			if (st.Kmodel3p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "DynamicGraphNetwork_model1" && len(va) > 1) { // net5.unit:982, g_structh.act:698
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "DynamicGraphNetwork_model2" && len(va) > 1) { // net5.unit:984, g_structh.act:698
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "GraphLearner_model1" && len(va) > 1) { // net5.unit:997, g_structh.act:698
		for _, st := range glob.Dats.ApGraphLearner {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "GraphLearner_model2" && len(va) > 1) { // net5.unit:999, g_structh.act:698
		for _, st := range glob.Dats.ApGraphLearner {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "NeuralProgram_model" && len(va) > 1) { // net5.unit:1017, g_structh.act:698
		for _, st := range glob.Dats.ApNeuralProgram {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ProgInstruction_model1" && len(va) > 1) { // net5.unit:1029, g_structh.act:698
		for _, st := range glob.Dats.ApProgInstruction {
			if (st.Kmodel1p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ProgInstruction_model2" && len(va) > 1) { // net5.unit:1031, g_structh.act:698
		for _, st := range glob.Dats.ApProgInstruction {
			if (st.Kmodel2p == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "LiquidNetwork_model" && len(va) > 1) { // net5.unit:1047, g_structh.act:698
		for _, st := range glob.Dats.ApLiquidNetwork {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // net5.unit:94, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Model > net5.unit:94, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Model > net5.unit:94, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpModel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Tensor" { // net5.unit:150, g_structh.act:676
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
	if va[0] == "Layer" { // net5.unit:164, g_structh.act:676
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
	if va[0] == "Config" { // net5.unit:633, g_structh.act:676
		for _, st := range me.ItsConfig {
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
	if va[0] == "ControlFlow" { // net5.unit:664, g_structh.act:676
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
	if va[0] == "GraphTensor" { // net5.unit:700, g_structh.act:676
		for _, st := range me.ItsGraphTensor {
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
	if va[0] == "ContinuousLayer" { // net5.unit:732, g_structh.act:676
		for _, st := range me.ItsContinuousLayer {
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
	if va[0] == "SearchSpace" { // net5.unit:753, g_structh.act:676
		for _, st := range me.ItsSearchSpace {
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
	if va[0] == "ArchitectureParam" { // net5.unit:787, g_structh.act:676
		for _, st := range me.ItsArchitectureParam {
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
	if va[0] == "ArchitectureGradient" { // net5.unit:798, g_structh.act:676
		for _, st := range me.ItsArchitectureGradient {
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
	if va[0] == "HyperNetwork" { // net5.unit:817, g_structh.act:676
		for _, st := range me.ItsHyperNetwork {
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
	if va[0] == "MemoryMatrix" { // net5.unit:848, g_structh.act:676
		for _, st := range me.ItsMemoryMatrix {
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
	if va[0] == "DifferentiableProgram" { // net5.unit:865, g_structh.act:676
		for _, st := range me.ItsDifferentiableProgram {
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
	if va[0] == "MetaLearner" { // net5.unit:891, g_structh.act:676
		for _, st := range me.ItsMetaLearner {
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
	if va[0] == "SparseExpertSystem" { // net5.unit:909, g_structh.act:676
		for _, st := range me.ItsSparseExpertSystem {
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
	if va[0] == "NeuralSDE" { // net5.unit:942, g_structh.act:676
		for _, st := range me.ItsNeuralSDE {
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
	if va[0] == "NeuralPDE" { // net5.unit:956, g_structh.act:676
		for _, st := range me.ItsNeuralPDE {
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
	if va[0] == "DynamicGraphNetwork" { // net5.unit:976, g_structh.act:676
		for _, st := range me.ItsDynamicGraphNetwork {
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
	if va[0] == "NeuralProgram" { // net5.unit:1009, g_structh.act:676
		for _, st := range me.ItsNeuralProgram {
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
	if va[0] == "LiquidNetwork" { // net5.unit:1041, g_structh.act:676
		for _, st := range me.ItsLiquidNetwork {
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
	if va[0] == "SymbolicShape" { // net5.unit:1058, g_structh.act:676
		for _, st := range me.ItsSymbolicShape {
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
	if va[0] == "JITCompiler" { // net5.unit:1069, g_structh.act:676
		for _, st := range me.ItsJITCompiler {
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
	if (va[0] == "Block_model") { // net5.unit:115, g_structh.act:583
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
	if (va[0] == "Arg_model") { // net5.unit:202, g_structh.act:583
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
	if (va[0] == "ExpertRoutingOp_model2") { // net5.unit:375, g_structh.act:583
		for _, st := range glob.Dats.ApExpertRoutingOp {
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
	if (va[0] == "Schedule_model1") { // net5.unit:650, g_structh.act:583
		for _, st := range glob.Dats.ApSchedule {
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
	if (va[0] == "Schedule_model2") { // net5.unit:653, g_structh.act:583
		for _, st := range glob.Dats.ApSchedule {
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
	if (va[0] == "Condition_model") { // net5.unit:680, g_structh.act:583
		for _, st := range glob.Dats.ApCondition {
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
	if (va[0] == "Branch_model") { // net5.unit:692, g_structh.act:583
		for _, st := range glob.Dats.ApBranch {
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
	if (va[0] == "StateTransfer_model1") { // net5.unit:722, g_structh.act:583
		for _, st := range glob.Dats.ApStateTransfer {
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
	if (va[0] == "StateTransfer_model2") { // net5.unit:724, g_structh.act:583
		for _, st := range glob.Dats.ApStateTransfer {
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
	if (va[0] == "ContinuousLayer_model") { // net5.unit:743, g_structh.act:583
		for _, st := range glob.Dats.ApContinuousLayer {
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
	if (va[0] == "SearchOp_model") { // net5.unit:769, g_structh.act:583
		for _, st := range glob.Dats.ApSearchOp {
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
	if (va[0] == "ArchitectureGradient_model") { // net5.unit:804, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureGradient {
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
	if (va[0] == "ArchitectureGradient_model2") { // net5.unit:806, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureGradient {
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
	if (va[0] == "HyperNetwork_model1") { // net5.unit:823, g_structh.act:583
		for _, st := range glob.Dats.ApHyperNetwork {
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
	if (va[0] == "HyperNetwork_model2") { // net5.unit:827, g_structh.act:583
		for _, st := range glob.Dats.ApHyperNetwork {
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
	if (va[0] == "WeightGenerator_model1") { // net5.unit:838, g_structh.act:583
		for _, st := range glob.Dats.ApWeightGenerator {
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
	if (va[0] == "WeightGenerator_model2") { // net5.unit:840, g_structh.act:583
		for _, st := range glob.Dats.ApWeightGenerator {
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
	if (va[0] == "ProgramInstruction_model1") { // net5.unit:883, g_structh.act:583
		for _, st := range glob.Dats.ApProgramInstruction {
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
	if (va[0] == "ProgramInstruction_model2") { // net5.unit:885, g_structh.act:583
		for _, st := range glob.Dats.ApProgramInstruction {
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
	if (va[0] == "MetaLearner_model1") { // net5.unit:897, g_structh.act:583
		for _, st := range glob.Dats.ApMetaLearner {
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
	if (va[0] == "MetaLearner_model2") { // net5.unit:899, g_structh.act:583
		for _, st := range glob.Dats.ApMetaLearner {
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
	if (va[0] == "SparseExpertSystem_model") { // net5.unit:915, g_structh.act:583
		for _, st := range glob.Dats.ApSparseExpertSystem {
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
	if (va[0] == "CapacityAwareRouter_model1") { // net5.unit:929, g_structh.act:583
		for _, st := range glob.Dats.ApCapacityAwareRouter {
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
	if (va[0] == "CapacityAwareRouter_model2") { // net5.unit:931, g_structh.act:583
		for _, st := range glob.Dats.ApCapacityAwareRouter {
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
	if (va[0] == "CapacityAwareRouter_model3") { // net5.unit:933, g_structh.act:583
		for _, st := range glob.Dats.ApCapacityAwareRouter {
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
	if (va[0] == "NeuralSDE_model1") { // net5.unit:948, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralSDE {
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
	if (va[0] == "NeuralSDE_model2") { // net5.unit:950, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralSDE {
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
	if (va[0] == "NeuralPDE_model1") { // net5.unit:963, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralPDE {
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
	if (va[0] == "NeuralPDE_model2") { // net5.unit:965, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralPDE {
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
	if (va[0] == "NeuralPDE_model3") { // net5.unit:967, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralPDE {
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
	if (va[0] == "DynamicGraphNetwork_model1") { // net5.unit:982, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
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
	if (va[0] == "DynamicGraphNetwork_model2") { // net5.unit:984, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
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
	if (va[0] == "GraphLearner_model1") { // net5.unit:997, g_structh.act:583
		for _, st := range glob.Dats.ApGraphLearner {
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
	if (va[0] == "GraphLearner_model2") { // net5.unit:999, g_structh.act:583
		for _, st := range glob.Dats.ApGraphLearner {
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
	if (va[0] == "NeuralProgram_model") { // net5.unit:1017, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralProgram {
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
	if (va[0] == "ProgInstruction_model1") { // net5.unit:1029, g_structh.act:583
		for _, st := range glob.Dats.ApProgInstruction {
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
	if (va[0] == "ProgInstruction_model2") { // net5.unit:1031, g_structh.act:583
		for _, st := range glob.Dats.ApProgInstruction {
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
	if (va[0] == "LiquidNetwork_model") { // net5.unit:1047, g_structh.act:583
		for _, st := range glob.Dats.ApLiquidNetwork {
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
	        fmt.Printf("?No its %s for Model %s,%s > net5.unit:94, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kmodelp int
	Klayersp int
	Kparametersp int
	ItsBlockParam [] *KpBlockParam 
	Childs [] Kp
}

func (me KpBlock) TypeName() string {
    return me.Comp
}
func (me KpBlock) GetLineNo() string {
	return me.LineNo
}

func loadBlock(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Klayersp = -1
	st.Kparametersp = -1
	name,_ := st.Names["block"]
	act.index["Block_" + name] = st.Me;
	st.MyName = name
	act.ApBlock = append(act.ApBlock, st)
	return 0
}

func (me KpBlock) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:115, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "layers" { // net5.unit:116, g_structh.act:609
		if (me.Klayersp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayersp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "parameters" { // net5.unit:117, g_structh.act:609
		if (me.Kparametersp >= 0 && len(va) > 1) {
			return( glob.Dats.ApBlockParam[ me.Kparametersp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "BlockInstance_block" && len(va) > 1) { // net5.unit:136, g_structh.act:698
		for _, st := range glob.Dats.ApBlockInstance {
			if (st.Kblockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // net5.unit:107, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBlock[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Block > net5.unit:107, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Block > net5.unit:107, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBlock) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "BlockParam" { // net5.unit:120, g_structh.act:676
		for _, st := range me.ItsBlockParam {
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
	if va[0] == "layers" {
		if me.Klayersp >= 0 {
			st := glob.Dats.ApLayer[ me.Klayersp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "parameters" {
		if me.Kparametersp >= 0 {
			st := glob.Dats.ApBlockParam[ me.Kparametersp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "BlockInstance_block") { // net5.unit:136, g_structh.act:583
		for _, st := range glob.Dats.ApBlockInstance {
			if (st.Kblockp == me.Me) {
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
	        fmt.Printf("?No its %s for Block %s,%s > net5.unit:107, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBlockParam struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpBlockParam) TypeName() string {
    return me.Comp
}
func (me KpBlockParam) GetLineNo() string {
	return me.LineNo
}

func loadBlockParam(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpBlockParam)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBlockParam)
	st.LineNo = lno
	st.Comp = "BlockParam";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApBlock ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " BlockParam has no Block parent\n") ;
		return 1
	}
	st.Parent = act.ApBlock[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " BlockParam under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApBlock[ len( act.ApBlock )-1 ].Childs = append(act.ApBlock[ len( act.ApBlock )-1 ].Childs, st)
	act.ApBlock[ len( act.ApBlock )-1 ].ItsBlockParam = append(act.ApBlock[ len( act.ApBlock )-1 ].ItsBlockParam, st)	// net5.unit:107, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_BlockParam_" + name	// net5.unit:125, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBlockParam = append(act.ApBlockParam, st)
	return 0
}

func (me KpBlockParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:107, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApBlock[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:120, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBlockParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,BlockParam > net5.unit:120, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,BlockParam > net5.unit:120, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBlockParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:107, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApBlock[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Block_parameters") { // net5.unit:117, g_structh.act:583
		for _, st := range glob.Dats.ApBlock {
			if (st.Kparametersp == me.Me) {
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
	        fmt.Printf("?No its %s for BlockParam %s,%s > net5.unit:120, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBlockInstance struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kblockp int
	Kparam_valuesp int
	ItsParamValue [] *KpParamValue 
	Childs [] Kp
}

func (me KpBlockInstance) TypeName() string {
    return me.Comp
}
func (me KpBlockInstance) GetLineNo() string {
	return me.LineNo
}

func loadBlockInstance(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpBlockInstance)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBlockInstance)
	st.LineNo = lno
	st.Comp = "BlockInstance";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kblockp = -1
	st.Kparam_valuesp = -1
	st.Kparentp = len( act.ApLayer ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " BlockInstance has no Layer parent\n") ;
		return 1
	}
	st.Parent = act.ApLayer[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " BlockInstance under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApLayer[ len( act.ApLayer )-1 ].Childs = append(act.ApLayer[ len( act.ApLayer )-1 ].Childs, st)
	act.ApLayer[ len( act.ApLayer )-1 ].ItsBlockInstance = append(act.ApLayer[ len( act.ApLayer )-1 ].ItsBlockInstance, st)	// net5.unit:164, g_structh.act:403
	name,_ := st.Names["instance"]
	s := strconv.Itoa(st.Kparentp) + "_BlockInstance_" + name	// net5.unit:135, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBlockInstance = append(act.ApBlockInstance, st)
	return 0
}

func (me KpBlockInstance) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "block" { // net5.unit:136, g_structh.act:609
		if (me.Kblockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApBlock[ me.Kblockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "param_values" { // net5.unit:137, g_structh.act:609
		if (me.Kparam_valuesp >= 0 && len(va) > 1) {
			return( glob.Dats.ApParamValue[ me.Kparam_valuesp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:164, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:130, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBlockInstance[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,BlockInstance > net5.unit:130, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,BlockInstance > net5.unit:130, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBlockInstance) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ParamValue" { // net5.unit:141, g_structh.act:676
		for _, st := range me.ItsParamValue {
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
	if va[0] == "parent" { // net5.unit:164, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApLayer[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "block" {
		if me.Kblockp >= 0 {
			st := glob.Dats.ApBlock[ me.Kblockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "param_values" {
		if me.Kparam_valuesp >= 0 {
			st := glob.Dats.ApParamValue[ me.Kparam_valuesp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for BlockInstance %s,%s > net5.unit:130, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpParamValue struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpParamValue) TypeName() string {
    return me.Comp
}
func (me KpParamValue) GetLineNo() string {
	return me.LineNo
}

func loadParamValue(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpParamValue)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApParamValue)
	st.LineNo = lno
	st.Comp = "ParamValue";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApBlockInstance ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ParamValue has no BlockInstance parent\n") ;
		return 1
	}
	st.Parent = act.ApBlockInstance[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ParamValue under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApBlockInstance[ len( act.ApBlockInstance )-1 ].Childs = append(act.ApBlockInstance[ len( act.ApBlockInstance )-1 ].Childs, st)
	act.ApBlockInstance[ len( act.ApBlockInstance )-1 ].ItsParamValue = append(act.ApBlockInstance[ len( act.ApBlockInstance )-1 ].ItsParamValue, st)	// net5.unit:130, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_ParamValue_" + name	// net5.unit:146, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApParamValue = append(act.ApParamValue, st)
	return 0
}

func (me KpParamValue) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:130, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApBlockInstance[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:141, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApParamValue[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ParamValue > net5.unit:141, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ParamValue > net5.unit:141, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpParamValue) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:130, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApBlockInstance[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "BlockInstance_param_values") { // net5.unit:137, g_structh.act:583
		for _, st := range glob.Dats.ApBlockInstance {
			if (st.Kparam_valuesp == me.Me) {
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
	        fmt.Printf("?No its %s for ParamValue %s,%s > net5.unit:141, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
	Kdtypep int
}

func (me KpTensor) TypeName() string {
    return me.Comp
}
func (me KpTensor) GetLineNo() string {
	return me.LineNo
}

func loadTensor(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kdtypep = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Tensor has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Tensor under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsTensor = append(act.ApModel[ len( act.ApModel )-1 ].ItsTensor, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["tensor"]
	s := strconv.Itoa(st.Kparentp) + "_Tensor_" + name	// net5.unit:155, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTensor = append(act.ApTensor, st)
	return 0
}

func (me KpTensor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "dtype" { // net5.unit:160, g_structh.act:609
		if (me.Kdtypep >= 0 && len(va) > 1) {
			return( glob.Dats.ApDtypeRule[ me.Kdtypep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:150, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Tensor > net5.unit:150, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Tensor > net5.unit:150, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTensor) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "dtype" {
		if me.Kdtypep >= 0 {
			st := glob.Dats.ApDtypeRule[ me.Kdtypep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Arg_tensor") { // net5.unit:203, g_structh.act:583
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
	if (va[0] == "Condition_input") { // net5.unit:681, g_structh.act:583
		for _, st := range glob.Dats.ApCondition {
			if (st.Kinputp == me.Me) {
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
	if (va[0] == "StateTransfer_source_state") { // net5.unit:723, g_structh.act:583
		for _, st := range glob.Dats.ApStateTransfer {
			if (st.Ksource_statep == me.Me) {
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
	if (va[0] == "StateTransfer_target_state") { // net5.unit:725, g_structh.act:583
		for _, st := range glob.Dats.ApStateTransfer {
			if (st.Ktarget_statep == me.Me) {
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
	if (va[0] == "HyperNetwork_condition") { // net5.unit:828, g_structh.act:583
		for _, st := range glob.Dats.ApHyperNetwork {
			if (st.Kconditionp == me.Me) {
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
	if (va[0] == "WeightGenerator_input_tensor") { // net5.unit:839, g_structh.act:583
		for _, st := range glob.Dats.ApWeightGenerator {
			if (st.Kinput_tensorp == me.Me) {
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
	if (va[0] == "WeightGenerator_output_tensor") { // net5.unit:841, g_structh.act:583
		for _, st := range glob.Dats.ApWeightGenerator {
			if (st.Koutput_tensorp == me.Me) {
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
	if (va[0] == "ProgramInstruction_operands") { // net5.unit:884, g_structh.act:583
		for _, st := range glob.Dats.ApProgramInstruction {
			if (st.Koperandsp == me.Me) {
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
	if (va[0] == "CapacityAwareRouter_gate_input") { // net5.unit:930, g_structh.act:583
		for _, st := range glob.Dats.ApCapacityAwareRouter {
			if (st.Kgate_inputp == me.Me) {
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
	if (va[0] == "CapacityAwareRouter_expert_mask") { // net5.unit:932, g_structh.act:583
		for _, st := range glob.Dats.ApCapacityAwareRouter {
			if (st.Kexpert_maskp == me.Me) {
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
	if (va[0] == "CapacityAwareRouter_token_assign") { // net5.unit:934, g_structh.act:583
		for _, st := range glob.Dats.ApCapacityAwareRouter {
			if (st.Ktoken_assignp == me.Me) {
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
	if (va[0] == "NeuralPDE_domain") { // net5.unit:964, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralPDE {
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
	if (va[0] == "NeuralPDE_boundary") { // net5.unit:966, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralPDE {
			if (st.Kboundaryp == me.Me) {
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
	if (va[0] == "NeuralPDE_initial") { // net5.unit:968, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralPDE {
			if (st.Kinitialp == me.Me) {
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
	if (va[0] == "NeuralProgram_memory") { // net5.unit:1018, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralProgram {
			if (st.Kmemoryp == me.Me) {
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
	if (va[0] == "ProgInstruction_operands") { // net5.unit:1030, g_structh.act:583
		for _, st := range glob.Dats.ApProgInstruction {
			if (st.Koperandsp == me.Me) {
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
	if (va[0] == "LiquidNetwork_time_const") { // net5.unit:1048, g_structh.act:583
		for _, st := range glob.Dats.ApLiquidNetwork {
			if (st.Ktime_constp == me.Me) {
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
	        fmt.Printf("?No its %s for Tensor %s,%s > net5.unit:150, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
	ItsBlockInstance [] *KpBlockInstance 
	ItsOp [] *KpOp 
	Childs [] Kp
}

func (me KpLayer) TypeName() string {
    return me.Comp
}
func (me KpLayer) GetLineNo() string {
	return me.LineNo
}

func loadLayer(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Layer has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Layer under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsLayer = append(act.ApModel[ len( act.ApModel )-1 ].ItsLayer, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["layer"]
	s := strconv.Itoa(st.Kparentp) + "_Layer_" + name	// net5.unit:169, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApLayer = append(act.ApLayer, st)
	return 0
}

func (me KpLayer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:164, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Layer > net5.unit:164, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Layer > net5.unit:164, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpLayer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "BlockInstance" { // net5.unit:130, g_structh.act:676
		for _, st := range me.ItsBlockInstance {
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
	if va[0] == "Op" { // net5.unit:178, g_structh.act:676
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Block_layers") { // net5.unit:116, g_structh.act:583
		for _, st := range glob.Dats.ApBlock {
			if (st.Klayersp == me.Me) {
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
	if (va[0] == "ExpertRoutingOp_experts") { // net5.unit:376, g_structh.act:583
		for _, st := range glob.Dats.ApExpertRoutingOp {
			if (st.Kexpertsp == me.Me) {
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
	if (va[0] == "Schedule_layer") { // net5.unit:651, g_structh.act:583
		for _, st := range glob.Dats.ApSchedule {
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
	if (va[0] == "Branch_layers") { // net5.unit:693, g_structh.act:583
		for _, st := range glob.Dats.ApBranch {
			if (st.Klayersp == me.Me) {
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
	if (va[0] == "ContinuousLayer_dynamics") { // net5.unit:744, g_structh.act:583
		for _, st := range glob.Dats.ApContinuousLayer {
			if (st.Kdynamicsp == me.Me) {
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
	if (va[0] == "SearchOp_layer") { // net5.unit:770, g_structh.act:583
		for _, st := range glob.Dats.ApSearchOp {
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
	if (va[0] == "ArchitectureGradient_layer") { // net5.unit:807, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureGradient {
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
	if (va[0] == "HyperNetwork_target_net") { // net5.unit:824, g_structh.act:583
		for _, st := range glob.Dats.ApHyperNetwork {
			if (st.Ktarget_netp == me.Me) {
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
	if (va[0] == "MetaLearner_inner_loop") { // net5.unit:898, g_structh.act:583
		for _, st := range glob.Dats.ApMetaLearner {
			if (st.Kinner_loopp == me.Me) {
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
	if (va[0] == "MetaLearner_outer_loop") { // net5.unit:900, g_structh.act:583
		for _, st := range glob.Dats.ApMetaLearner {
			if (st.Kouter_loopp == me.Me) {
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
	if (va[0] == "SparseExpertSystem_experts") { // net5.unit:916, g_structh.act:583
		for _, st := range glob.Dats.ApSparseExpertSystem {
			if (st.Kexpertsp == me.Me) {
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
	if (va[0] == "NeuralSDE_drift_net") { // net5.unit:949, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralSDE {
			if (st.Kdrift_netp == me.Me) {
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
	if (va[0] == "NeuralSDE_diffusion_net") { // net5.unit:951, g_structh.act:583
		for _, st := range glob.Dats.ApNeuralSDE {
			if (st.Kdiffusion_netp == me.Me) {
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
	if (va[0] == "DynamicGraphNetwork_edge_predictor") { // net5.unit:985, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
			if (st.Kedge_predictorp == me.Me) {
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
	        fmt.Printf("?No its %s for Layer %s,%s > net5.unit:164, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
	Kop_typep int
	Kkernelp int
	Kkernel_opp int
	ItsArg [] *KpArg 
	ItsAttentionOp [] *KpAttentionOp 
	ItsAttentionFreeOp [] *KpAttentionFreeOp 
	ItsGraphOp [] *KpGraphOp 
	ItsGraphLearningOp [] *KpGraphLearningOp 
	ItsStatefulOp [] *KpStatefulOp 
	ItsLTCOp [] *KpLTCOp 
	ItsODEOp [] *KpODEOp 
	ItsSDEOp [] *KpSDEOp 
	ItsPDEOp [] *KpPDEOp 
	ItsMemoryOp [] *KpMemoryOp 
	ItsExpertRoutingOp [] *KpExpertRoutingOp 
	ItsSamplingOp [] *KpSamplingOp 
	ItsStochasticOp [] *KpStochasticOp 
	ItsDynamicRoutingOp [] *KpDynamicRoutingOp 
	ItsDynamicOp [] *KpDynamicOp 
	ItsNeuralProgramOp [] *KpNeuralProgramOp 
	ItsWeightGenerationOp [] *KpWeightGenerationOp 
	ItsArchitectureSearchOp [] *KpArchitectureSearchOp 
	ItsContinuousDepthOp [] *KpContinuousDepthOp 
	Childs [] Kp
}

func (me KpOp) TypeName() string {
    return me.Comp
}
func (me KpOp) GetLineNo() string {
	return me.LineNo
}

func loadOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kop_typep = -1
	st.Kkernelp = -1
	st.Kkernel_opp = -1
	st.Kparentp = len( act.ApLayer ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Op has no Layer parent\n") ;
		return 1
	}
	st.Parent = act.ApLayer[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Op under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApLayer[ len( act.ApLayer )-1 ].Childs = append(act.ApLayer[ len( act.ApLayer )-1 ].Childs, st)
	act.ApLayer[ len( act.ApLayer )-1 ].ItsOp = append(act.ApLayer[ len( act.ApLayer )-1 ].ItsOp, st)	// net5.unit:164, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_Op_" + name	// net5.unit:184, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOp = append(act.ApOp, st)
	return 0
}

func (me KpOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "op_type" { // net5.unit:185, g_structh.act:619
		if (me.Kop_typep >= 0 && len(va) > 1) {
			return( me.Childs[ me.Kop_typep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel" { // net5.unit:190, g_structh.act:609
		if (me.Kkernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kkernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel_op" { // net5.unit:191, g_structh.act:609
		if (me.Kkernel_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernelOp[ me.Kkernel_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:164, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:178, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Op > net5.unit:178, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Op > net5.unit:178, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Arg" { // net5.unit:194, g_structh.act:676
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
	if va[0] == "AttentionOp" { // net5.unit:212, g_structh.act:676
		for _, st := range me.ItsAttentionOp {
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
	if va[0] == "AttentionFreeOp" { // net5.unit:224, g_structh.act:676
		for _, st := range me.ItsAttentionFreeOp {
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
	if va[0] == "GraphOp" { // net5.unit:251, g_structh.act:676
		for _, st := range me.ItsGraphOp {
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
	if va[0] == "GraphLearningOp" { // net5.unit:262, g_structh.act:676
		for _, st := range me.ItsGraphLearningOp {
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
	if va[0] == "StatefulOp" { // net5.unit:279, g_structh.act:676
		for _, st := range me.ItsStatefulOp {
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
	if va[0] == "LTCOp" { // net5.unit:290, g_structh.act:676
		for _, st := range me.ItsLTCOp {
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
	if va[0] == "ODEOp" { // net5.unit:307, g_structh.act:676
		for _, st := range me.ItsODEOp {
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
	if va[0] == "SDEOp" { // net5.unit:321, g_structh.act:676
		for _, st := range me.ItsSDEOp {
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
	if va[0] == "PDEOp" { // net5.unit:333, g_structh.act:676
		for _, st := range me.ItsPDEOp {
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
	if va[0] == "MemoryOp" { // net5.unit:348, g_structh.act:676
		for _, st := range me.ItsMemoryOp {
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
	if va[0] == "ExpertRoutingOp" { // net5.unit:366, g_structh.act:676
		for _, st := range me.ItsExpertRoutingOp {
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
	if va[0] == "SamplingOp" { // net5.unit:395, g_structh.act:676
		for _, st := range me.ItsSamplingOp {
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
	if va[0] == "StochasticOp" { // net5.unit:406, g_structh.act:676
		for _, st := range me.ItsStochasticOp {
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
	if va[0] == "DynamicRoutingOp" { // net5.unit:420, g_structh.act:676
		for _, st := range me.ItsDynamicRoutingOp {
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
	if va[0] == "DynamicOp" { // net5.unit:430, g_structh.act:676
		for _, st := range me.ItsDynamicOp {
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
	if va[0] == "NeuralProgramOp" { // net5.unit:443, g_structh.act:676
		for _, st := range me.ItsNeuralProgramOp {
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
	if va[0] == "WeightGenerationOp" { // net5.unit:458, g_structh.act:676
		for _, st := range me.ItsWeightGenerationOp {
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
	if va[0] == "ArchitectureSearchOp" { // net5.unit:473, g_structh.act:676
		for _, st := range me.ItsArchitectureSearchOp {
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
	if va[0] == "ContinuousDepthOp" { // net5.unit:488, g_structh.act:676
		for _, st := range me.ItsContinuousDepthOp {
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
	if va[0] == "parent" { // net5.unit:164, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApLayer[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "op_type" {
		if me.Kop_typep >= 0 {
			st := me.Childs[ me.Kop_typep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "kernel" {
		if me.Kkernelp >= 0 {
			st := glob.Dats.ApKernel[ me.Kkernelp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "kernel_op" {
		if me.Kkernel_opp >= 0 {
			st := glob.Dats.ApKernelOp[ me.Kkernel_opp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Schedule_op") { // net5.unit:652, g_structh.act:583
		for _, st := range glob.Dats.ApSchedule {
			if (st.Kopp == me.Me) {
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
	if (va[0] == "SearchOp_operation") { // net5.unit:771, g_structh.act:583
		for _, st := range glob.Dats.ApSearchOp {
			if (st.Koperationp == me.Me) {
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
	if (va[0] == "ArchitectureGradient_gradient_ops") { // net5.unit:808, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureGradient {
			if (st.Kgradient_opsp == me.Me) {
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
	        fmt.Printf("?No its %s for Op %s,%s > net5.unit:178, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
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

func loadArg(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Arg under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsArg = append(act.ApOp[ len( act.ApOp )-1 ].ItsArg, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["arg"]
	s := strconv.Itoa(st.Kparentp) + "_Arg_" + name	// net5.unit:200, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArg = append(act.ApArg, st)
	return 0
}

func (me KpArg) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:202, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "tensor" { // net5.unit:203, g_structh.act:609
		if (me.Ktensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:194, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArg[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Arg > net5.unit:194, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Arg > net5.unit:194, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArg) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
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
	        fmt.Printf("?No its %s for Arg %s,%s > net5.unit:194, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAttentionOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpAttentionOp) TypeName() string {
    return me.Comp
}
func (me KpAttentionOp) GetLineNo() string {
	return me.LineNo
}

func loadAttentionOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAttentionOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAttentionOp)
	st.LineNo = lno
	st.Comp = "AttentionOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AttentionOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AttentionOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsAttentionOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsAttentionOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["attn_op"]
	s := strconv.Itoa(st.Kparentp) + "_AttentionOp_" + name	// net5.unit:218, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApAttentionOp = append(act.ApAttentionOp, st)
	return 0
}

func (me KpAttentionOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:212, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAttentionOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AttentionOp > net5.unit:212, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AttentionOp > net5.unit:212, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAttentionOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for AttentionOp %s,%s > net5.unit:212, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAttentionFreeOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsRecurrentFormulation [] *KpRecurrentFormulation 
	Childs [] Kp
}

func (me KpAttentionFreeOp) TypeName() string {
    return me.Comp
}
func (me KpAttentionFreeOp) GetLineNo() string {
	return me.LineNo
}

func loadAttentionFreeOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAttentionFreeOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAttentionFreeOp)
	st.LineNo = lno
	st.Comp = "AttentionFreeOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AttentionFreeOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AttentionFreeOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsAttentionFreeOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsAttentionFreeOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["aft_op"]
	s := strconv.Itoa(st.Kparentp) + "_AttentionFreeOp_" + name	// net5.unit:230, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApAttentionFreeOp = append(act.ApAttentionFreeOp, st)
	return 0
}

func (me KpAttentionFreeOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:224, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAttentionFreeOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AttentionFreeOp > net5.unit:224, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AttentionFreeOp > net5.unit:224, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAttentionFreeOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "RecurrentFormulation" { // net5.unit:236, g_structh.act:676
		for _, st := range me.ItsRecurrentFormulation {
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
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for AttentionFreeOp %s,%s > net5.unit:224, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpRecurrentFormulation struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpRecurrentFormulation) TypeName() string {
    return me.Comp
}
func (me KpRecurrentFormulation) GetLineNo() string {
	return me.LineNo
}

func loadRecurrentFormulation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpRecurrentFormulation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApRecurrentFormulation)
	st.LineNo = lno
	st.Comp = "RecurrentFormulation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApAttentionFreeOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " RecurrentFormulation has no AttentionFreeOp parent\n") ;
		return 1
	}
	st.Parent = act.ApAttentionFreeOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " RecurrentFormulation under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApAttentionFreeOp[ len( act.ApAttentionFreeOp )-1 ].Childs = append(act.ApAttentionFreeOp[ len( act.ApAttentionFreeOp )-1 ].Childs, st)
	act.ApAttentionFreeOp[ len( act.ApAttentionFreeOp )-1 ].ItsRecurrentFormulation = append(act.ApAttentionFreeOp[ len( act.ApAttentionFreeOp )-1 ].ItsRecurrentFormulation, st)	// net5.unit:224, g_structh.act:403
	name,_ := st.Names["formulation"]
	s := strconv.Itoa(st.Kparentp) + "_RecurrentFormulation_" + name	// net5.unit:241, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApRecurrentFormulation = append(act.ApRecurrentFormulation, st)
	return 0
}

func (me KpRecurrentFormulation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:224, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApAttentionFreeOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:236, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApRecurrentFormulation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,RecurrentFormulation > net5.unit:236, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,RecurrentFormulation > net5.unit:236, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpRecurrentFormulation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:224, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApAttentionFreeOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for RecurrentFormulation %s,%s > net5.unit:236, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpGraphOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpGraphOp) TypeName() string {
    return me.Comp
}
func (me KpGraphOp) GetLineNo() string {
	return me.LineNo
}

func loadGraphOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpGraphOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApGraphOp)
	st.LineNo = lno
	st.Comp = "GraphOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " GraphOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " GraphOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsGraphOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsGraphOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["graph_op"]
	s := strconv.Itoa(st.Kparentp) + "_GraphOp_" + name	// net5.unit:257, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApGraphOp = append(act.ApGraphOp, st)
	return 0
}

func (me KpGraphOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:251, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApGraphOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,GraphOp > net5.unit:251, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,GraphOp > net5.unit:251, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpGraphOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for GraphOp %s,%s > net5.unit:251, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpGraphLearningOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpGraphLearningOp) TypeName() string {
    return me.Comp
}
func (me KpGraphLearningOp) GetLineNo() string {
	return me.LineNo
}

func loadGraphLearningOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpGraphLearningOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApGraphLearningOp)
	st.LineNo = lno
	st.Comp = "GraphLearningOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " GraphLearningOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " GraphLearningOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsGraphLearningOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsGraphLearningOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["graph_op"]
	s := strconv.Itoa(st.Kparentp) + "_GraphLearningOp_" + name	// net5.unit:268, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApGraphLearningOp = append(act.ApGraphLearningOp, st)
	return 0
}

func (me KpGraphLearningOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:262, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApGraphLearningOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,GraphLearningOp > net5.unit:262, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,GraphLearningOp > net5.unit:262, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpGraphLearningOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for GraphLearningOp %s,%s > net5.unit:262, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpStatefulOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpStatefulOp) TypeName() string {
    return me.Comp
}
func (me KpStatefulOp) GetLineNo() string {
	return me.LineNo
}

func loadStatefulOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpStatefulOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApStatefulOp)
	st.LineNo = lno
	st.Comp = "StatefulOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " StatefulOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " StatefulOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsStatefulOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsStatefulOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["state_op"]
	s := strconv.Itoa(st.Kparentp) + "_StatefulOp_" + name	// net5.unit:285, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApStatefulOp = append(act.ApStatefulOp, st)
	return 0
}

func (me KpStatefulOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:279, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApStatefulOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,StatefulOp > net5.unit:279, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,StatefulOp > net5.unit:279, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpStatefulOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for StatefulOp %s,%s > net5.unit:279, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpLTCOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpLTCOp) TypeName() string {
    return me.Comp
}
func (me KpLTCOp) GetLineNo() string {
	return me.LineNo
}

func loadLTCOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpLTCOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApLTCOp)
	st.LineNo = lno
	st.Comp = "LTCOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " LTCOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " LTCOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsLTCOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsLTCOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["ltc_op"]
	s := strconv.Itoa(st.Kparentp) + "_LTCOp_" + name	// net5.unit:296, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApLTCOp = append(act.ApLTCOp, st)
	return 0
}

func (me KpLTCOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:290, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLTCOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,LTCOp > net5.unit:290, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,LTCOp > net5.unit:290, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpLTCOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for LTCOp %s,%s > net5.unit:290, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpODEOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpODEOp) TypeName() string {
    return me.Comp
}
func (me KpODEOp) GetLineNo() string {
	return me.LineNo
}

func loadODEOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpODEOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApODEOp)
	st.LineNo = lno
	st.Comp = "ODEOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ODEOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ODEOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsODEOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsODEOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["ode_op"]
	s := strconv.Itoa(st.Kparentp) + "_ODEOp_" + name	// net5.unit:313, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApODEOp = append(act.ApODEOp, st)
	return 0
}

func (me KpODEOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:307, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApODEOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ODEOp > net5.unit:307, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ODEOp > net5.unit:307, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpODEOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ODEOp %s,%s > net5.unit:307, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSDEOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpSDEOp) TypeName() string {
    return me.Comp
}
func (me KpSDEOp) GetLineNo() string {
	return me.LineNo
}

func loadSDEOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSDEOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSDEOp)
	st.LineNo = lno
	st.Comp = "SDEOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SDEOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SDEOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsSDEOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsSDEOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["sde_op"]
	s := strconv.Itoa(st.Kparentp) + "_SDEOp_" + name	// net5.unit:327, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSDEOp = append(act.ApSDEOp, st)
	return 0
}

func (me KpSDEOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:321, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSDEOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SDEOp > net5.unit:321, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SDEOp > net5.unit:321, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSDEOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SDEOp %s,%s > net5.unit:321, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPDEOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpPDEOp) TypeName() string {
    return me.Comp
}
func (me KpPDEOp) GetLineNo() string {
	return me.LineNo
}

func loadPDEOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPDEOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPDEOp)
	st.LineNo = lno
	st.Comp = "PDEOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PDEOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PDEOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsPDEOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsPDEOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["pde_op"]
	s := strconv.Itoa(st.Kparentp) + "_PDEOp_" + name	// net5.unit:339, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApPDEOp = append(act.ApPDEOp, st)
	return 0
}

func (me KpPDEOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:333, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPDEOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PDEOp > net5.unit:333, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PDEOp > net5.unit:333, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPDEOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PDEOp %s,%s > net5.unit:333, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMemoryOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpMemoryOp) TypeName() string {
    return me.Comp
}
func (me KpMemoryOp) GetLineNo() string {
	return me.LineNo
}

func loadMemoryOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMemoryOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMemoryOp)
	st.LineNo = lno
	st.Comp = "MemoryOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MemoryOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MemoryOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsMemoryOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsMemoryOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["mem_op"]
	s := strconv.Itoa(st.Kparentp) + "_MemoryOp_" + name	// net5.unit:354, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApMemoryOp = append(act.ApMemoryOp, st)
	return 0
}

func (me KpMemoryOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:348, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMemoryOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MemoryOp > net5.unit:348, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MemoryOp > net5.unit:348, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMemoryOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MemoryOp %s,%s > net5.unit:348, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpExpertRoutingOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel2p int
	Kexpertsp int
	ItsCapacityAwareRoutingOp [] *KpCapacityAwareRoutingOp 
	Childs [] Kp
}

func (me KpExpertRoutingOp) TypeName() string {
    return me.Comp
}
func (me KpExpertRoutingOp) GetLineNo() string {
	return me.LineNo
}

func loadExpertRoutingOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpExpertRoutingOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApExpertRoutingOp)
	st.LineNo = lno
	st.Comp = "ExpertRoutingOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel2p = -1
	st.Kexpertsp = -1
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ExpertRoutingOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ExpertRoutingOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsExpertRoutingOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsExpertRoutingOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["expert_op"]
	s := strconv.Itoa(st.Kparentp) + "_ExpertRoutingOp_" + name	// net5.unit:372, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApExpertRoutingOp = append(act.ApExpertRoutingOp, st)
	return 0
}

func (me KpExpertRoutingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model2" { // net5.unit:375, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "experts" { // net5.unit:376, g_structh.act:609
		if (me.Kexpertsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kexpertsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:366, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExpertRoutingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ExpertRoutingOp > net5.unit:366, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ExpertRoutingOp > net5.unit:366, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExpertRoutingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "CapacityAwareRoutingOp" { // net5.unit:382, g_structh.act:676
		for _, st := range me.ItsCapacityAwareRoutingOp {
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
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
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
	if va[0] == "experts" {
		if me.Kexpertsp >= 0 {
			st := glob.Dats.ApLayer[ me.Kexpertsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ExpertRoutingOp %s,%s > net5.unit:366, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCapacityAwareRoutingOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpCapacityAwareRoutingOp) TypeName() string {
    return me.Comp
}
func (me KpCapacityAwareRoutingOp) GetLineNo() string {
	return me.LineNo
}

func loadCapacityAwareRoutingOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCapacityAwareRoutingOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCapacityAwareRoutingOp)
	st.LineNo = lno
	st.Comp = "CapacityAwareRoutingOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApExpertRoutingOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " CapacityAwareRoutingOp has no ExpertRoutingOp parent\n") ;
		return 1
	}
	st.Parent = act.ApExpertRoutingOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " CapacityAwareRoutingOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApExpertRoutingOp[ len( act.ApExpertRoutingOp )-1 ].Childs = append(act.ApExpertRoutingOp[ len( act.ApExpertRoutingOp )-1 ].Childs, st)
	act.ApExpertRoutingOp[ len( act.ApExpertRoutingOp )-1 ].ItsCapacityAwareRoutingOp = append(act.ApExpertRoutingOp[ len( act.ApExpertRoutingOp )-1 ].ItsCapacityAwareRoutingOp, st)	// net5.unit:366, g_structh.act:403
	name,_ := st.Names["router"]
	s := strconv.Itoa(st.Kparentp) + "_CapacityAwareRoutingOp_" + name	// net5.unit:388, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCapacityAwareRoutingOp = append(act.ApCapacityAwareRoutingOp, st)
	return 0
}

func (me KpCapacityAwareRoutingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:366, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExpertRoutingOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:382, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCapacityAwareRoutingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CapacityAwareRoutingOp > net5.unit:382, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CapacityAwareRoutingOp > net5.unit:382, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCapacityAwareRoutingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:366, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApExpertRoutingOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CapacityAwareRoutingOp %s,%s > net5.unit:382, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSamplingOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpSamplingOp) TypeName() string {
    return me.Comp
}
func (me KpSamplingOp) GetLineNo() string {
	return me.LineNo
}

func loadSamplingOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSamplingOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSamplingOp)
	st.LineNo = lno
	st.Comp = "SamplingOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SamplingOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SamplingOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsSamplingOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsSamplingOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["sampling_op"]
	s := strconv.Itoa(st.Kparentp) + "_SamplingOp_" + name	// net5.unit:401, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSamplingOp = append(act.ApSamplingOp, st)
	return 0
}

func (me KpSamplingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:395, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSamplingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SamplingOp > net5.unit:395, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SamplingOp > net5.unit:395, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSamplingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SamplingOp %s,%s > net5.unit:395, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpStochasticOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpStochasticOp) TypeName() string {
    return me.Comp
}
func (me KpStochasticOp) GetLineNo() string {
	return me.LineNo
}

func loadStochasticOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpStochasticOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApStochasticOp)
	st.LineNo = lno
	st.Comp = "StochasticOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " StochasticOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " StochasticOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsStochasticOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsStochasticOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["Stochastic_op"]
	s := strconv.Itoa(st.Kparentp) + "_StochasticOp_" + name	// net5.unit:412, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApStochasticOp = append(act.ApStochasticOp, st)
	return 0
}

func (me KpStochasticOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:406, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApStochasticOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,StochasticOp > net5.unit:406, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,StochasticOp > net5.unit:406, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpStochasticOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for StochasticOp %s,%s > net5.unit:406, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDynamicRoutingOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpDynamicRoutingOp) TypeName() string {
    return me.Comp
}
func (me KpDynamicRoutingOp) GetLineNo() string {
	return me.LineNo
}

func loadDynamicRoutingOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDynamicRoutingOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDynamicRoutingOp)
	st.LineNo = lno
	st.Comp = "DynamicRoutingOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DynamicRoutingOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DynamicRoutingOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsDynamicRoutingOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsDynamicRoutingOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["dr_op"]
	s := strconv.Itoa(st.Kparentp) + "_DynamicRoutingOp_" + name	// net5.unit:426, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDynamicRoutingOp = append(act.ApDynamicRoutingOp, st)
	return 0
}

func (me KpDynamicRoutingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:420, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicRoutingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DynamicRoutingOp > net5.unit:420, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DynamicRoutingOp > net5.unit:420, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDynamicRoutingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DynamicRoutingOp %s,%s > net5.unit:420, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDynamicOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpDynamicOp) TypeName() string {
    return me.Comp
}
func (me KpDynamicOp) GetLineNo() string {
	return me.LineNo
}

func loadDynamicOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDynamicOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDynamicOp)
	st.LineNo = lno
	st.Comp = "DynamicOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DynamicOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DynamicOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsDynamicOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsDynamicOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["dynamic_op"]
	s := strconv.Itoa(st.Kparentp) + "_DynamicOp_" + name	// net5.unit:436, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDynamicOp = append(act.ApDynamicOp, st)
	return 0
}

func (me KpDynamicOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:430, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DynamicOp > net5.unit:430, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DynamicOp > net5.unit:430, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDynamicOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DynamicOp %s,%s > net5.unit:430, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuralProgramOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpNeuralProgramOp) TypeName() string {
    return me.Comp
}
func (me KpNeuralProgramOp) GetLineNo() string {
	return me.LineNo
}

func loadNeuralProgramOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuralProgramOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuralProgramOp)
	st.LineNo = lno
	st.Comp = "NeuralProgramOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralProgramOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralProgramOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsNeuralProgramOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsNeuralProgramOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["np_op"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralProgramOp_" + name	// net5.unit:449, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralProgramOp = append(act.ApNeuralProgramOp, st)
	return 0
}

func (me KpNeuralProgramOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:443, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralProgramOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralProgramOp > net5.unit:443, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralProgramOp > net5.unit:443, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralProgramOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralProgramOp %s,%s > net5.unit:443, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpWeightGenerationOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpWeightGenerationOp) TypeName() string {
    return me.Comp
}
func (me KpWeightGenerationOp) GetLineNo() string {
	return me.LineNo
}

func loadWeightGenerationOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpWeightGenerationOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApWeightGenerationOp)
	st.LineNo = lno
	st.Comp = "WeightGenerationOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " WeightGenerationOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " WeightGenerationOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsWeightGenerationOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsWeightGenerationOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["wg_op"]
	s := strconv.Itoa(st.Kparentp) + "_WeightGenerationOp_" + name	// net5.unit:464, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApWeightGenerationOp = append(act.ApWeightGenerationOp, st)
	return 0
}

func (me KpWeightGenerationOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:458, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApWeightGenerationOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,WeightGenerationOp > net5.unit:458, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,WeightGenerationOp > net5.unit:458, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpWeightGenerationOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for WeightGenerationOp %s,%s > net5.unit:458, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpArchitectureSearchOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpArchitectureSearchOp) TypeName() string {
    return me.Comp
}
func (me KpArchitectureSearchOp) GetLineNo() string {
	return me.LineNo
}

func loadArchitectureSearchOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpArchitectureSearchOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApArchitectureSearchOp)
	st.LineNo = lno
	st.Comp = "ArchitectureSearchOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ArchitectureSearchOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ArchitectureSearchOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsArchitectureSearchOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsArchitectureSearchOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["as_op"]
	s := strconv.Itoa(st.Kparentp) + "_ArchitectureSearchOp_" + name	// net5.unit:479, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArchitectureSearchOp = append(act.ApArchitectureSearchOp, st)
	return 0
}

func (me KpArchitectureSearchOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:473, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureSearchOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArchitectureSearchOp > net5.unit:473, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArchitectureSearchOp > net5.unit:473, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitectureSearchOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ArchitectureSearchOp %s,%s > net5.unit:473, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpContinuousDepthOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpContinuousDepthOp) TypeName() string {
    return me.Comp
}
func (me KpContinuousDepthOp) GetLineNo() string {
	return me.LineNo
}

func loadContinuousDepthOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpContinuousDepthOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApContinuousDepthOp)
	st.LineNo = lno
	st.Comp = "ContinuousDepthOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ContinuousDepthOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ContinuousDepthOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsContinuousDepthOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsContinuousDepthOp, st)	// net5.unit:178, g_structh.act:403
	name,_ := st.Names["cd_op"]
	s := strconv.Itoa(st.Kparentp) + "_ContinuousDepthOp_" + name	// net5.unit:494, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApContinuousDepthOp = append(act.ApContinuousDepthOp, st)
	return 0
}

func (me KpContinuousDepthOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:178, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:488, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApContinuousDepthOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ContinuousDepthOp > net5.unit:488, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ContinuousDepthOp > net5.unit:488, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpContinuousDepthOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:178, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ContinuousDepthOp %s,%s > net5.unit:488, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOpTypeRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpOpTypeRule) TypeName() string {
    return me.Comp
}
func (me KpOpTypeRule) GetLineNo() string {
	return me.LineNo
}

func loadOpTypeRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOpTypeRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOpTypeRule)
	st.LineNo = lno
	st.Comp = "OpTypeRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["op_type"]
	act.index["OpTypeRule_" + name] = st.Me;
	st.MyName = name
	act.ApOpTypeRule = append(act.ApOpTypeRule, st)
	return 0
}

func (me KpOpTypeRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:506, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOpTypeRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OpTypeRule > net5.unit:506, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OpTypeRule > net5.unit:506, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOpTypeRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for OpTypeRule %s,%s > net5.unit:506, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpArgRoleRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kdtypep int
}

func (me KpArgRoleRule) TypeName() string {
    return me.Comp
}
func (me KpArgRoleRule) GetLineNo() string {
	return me.LineNo
}

func loadArgRoleRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpArgRoleRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApArgRoleRule)
	st.LineNo = lno
	st.Comp = "ArgRoleRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kdtypep = -1
	name,_ := st.Names["role"]
	act.index["ArgRoleRule_" + name] = st.Me;
	st.MyName = name
	act.ApArgRoleRule = append(act.ApArgRoleRule, st)
	return 0
}

func (me KpArgRoleRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "dtype" { // net5.unit:629, g_structh.act:609
		if (me.Kdtypep >= 0 && len(va) > 1) {
			return( glob.Dats.ApDtypeRule[ me.Kdtypep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // net5.unit:538, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArgRoleRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArgRoleRule > net5.unit:538, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArgRoleRule > net5.unit:538, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArgRoleRule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "dtype" {
		if me.Kdtypep >= 0 {
			st := glob.Dats.ApDtypeRule[ me.Kdtypep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ArgRoleRule %s,%s > net5.unit:538, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
	Ktargetp int
	Kopt_flagsp int
	ItsSchedule [] *KpSchedule 
	ItsStateTransfer [] *KpStateTransfer 
	Childs [] Kp
}

func (me KpConfig) TypeName() string {
    return me.Comp
}
func (me KpConfig) GetLineNo() string {
	return me.LineNo
}

func loadConfig(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Ktargetp = -1
	st.Kopt_flagsp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Config has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Config under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsConfig = append(act.ApModel[ len( act.ApModel )-1 ].ItsConfig, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["config"]
	s := strconv.Itoa(st.Kparentp) + "_Config_" + name	// net5.unit:638, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApConfig = append(act.ApConfig, st)
	return 0
}

func (me KpConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target" { // net5.unit:639, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTargetRule[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "opt_flags" { // net5.unit:641, g_structh.act:609
		if (me.Kopt_flagsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFlagRule[ me.Kopt_flagsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:633, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Config > net5.unit:633, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Config > net5.unit:633, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Schedule" { // net5.unit:644, g_structh.act:676
		for _, st := range me.ItsSchedule {
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
	if va[0] == "StateTransfer" { // net5.unit:715, g_structh.act:676
		for _, st := range me.ItsStateTransfer {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "target" {
		if me.Ktargetp >= 0 {
			st := glob.Dats.ApTargetRule[ me.Ktargetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "opt_flags" {
		if me.Kopt_flagsp >= 0 {
			st := glob.Dats.ApFlagRule[ me.Kopt_flagsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Config %s,%s > net5.unit:633, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSchedule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Klayerp int
	Kopp int
	Kmodel2p int
	Kcontrolp int
}

func (me KpSchedule) TypeName() string {
    return me.Comp
}
func (me KpSchedule) GetLineNo() string {
	return me.LineNo
}

func loadSchedule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSchedule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSchedule)
	st.LineNo = lno
	st.Comp = "Schedule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Klayerp = -1
	st.Kopp = -1
	st.Kmodel2p = -1
	st.Kcontrolp = -1
	st.Kparentp = len( act.ApConfig ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Schedule has no Config parent\n") ;
		return 1
	}
	st.Parent = act.ApConfig[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Schedule under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApConfig[ len( act.ApConfig )-1 ].Childs = append(act.ApConfig[ len( act.ApConfig )-1 ].Childs, st)
	act.ApConfig[ len( act.ApConfig )-1 ].ItsSchedule = append(act.ApConfig[ len( act.ApConfig )-1 ].ItsSchedule, st)	// net5.unit:633, g_structh.act:403
	act.ApSchedule = append(act.ApSchedule, st)
	return 0
}

func (me KpSchedule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:650, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "layer" { // net5.unit:651, g_structh.act:609
		if (me.Klayerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "op" { // net5.unit:652, g_structh.act:609
		if (me.Kopp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kopp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:653, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "control" { // net5.unit:654, g_structh.act:609
		if (me.Kcontrolp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kcontrolp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:633, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:644, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSchedule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Schedule > net5.unit:644, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Schedule > net5.unit:644, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSchedule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:633, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApConfig[ me.Kparentp ]
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
	if va[0] == "op" {
		if me.Kopp >= 0 {
			st := glob.Dats.ApOp[ me.Kopp ]
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
	if va[0] == "control" {
		if me.Kcontrolp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kcontrolp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Schedule %s,%s > net5.unit:644, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
	ItsCondition [] *KpCondition 
	ItsBranch [] *KpBranch 
	Childs [] Kp
}

func (me KpControlFlow) TypeName() string {
    return me.Comp
}
func (me KpControlFlow) GetLineNo() string {
	return me.LineNo
}

func loadControlFlow(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ControlFlow has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ControlFlow under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsControlFlow = append(act.ApModel[ len( act.ApModel )-1 ].ItsControlFlow, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["control"]
	s := strconv.Itoa(st.Kparentp) + "_ControlFlow_" + name	// net5.unit:669, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApControlFlow = append(act.ApControlFlow, st)
	return 0
}

func (me KpControlFlow) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:664, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ControlFlow > net5.unit:664, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ControlFlow > net5.unit:664, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpControlFlow) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Condition" { // net5.unit:673, g_structh.act:676
		for _, st := range me.ItsCondition {
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
	if va[0] == "Branch" { // net5.unit:685, g_structh.act:676
		for _, st := range me.ItsBranch {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Schedule_control") { // net5.unit:654, g_structh.act:583
		for _, st := range glob.Dats.ApSchedule {
			if (st.Kcontrolp == me.Me) {
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
	if (va[0] == "ProgramInstruction_control_flow") { // net5.unit:886, g_structh.act:583
		for _, st := range glob.Dats.ApProgramInstruction {
			if (st.Kcontrol_flowp == me.Me) {
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
	if (va[0] == "ProgInstruction_control_flow") { // net5.unit:1032, g_structh.act:583
		for _, st := range glob.Dats.ApProgInstruction {
			if (st.Kcontrol_flowp == me.Me) {
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
	        fmt.Printf("?No its %s for ControlFlow %s,%s > net5.unit:664, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCondition struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Kinputp int
}

func (me KpCondition) TypeName() string {
    return me.Comp
}
func (me KpCondition) GetLineNo() string {
	return me.LineNo
}

func loadCondition(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCondition)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCondition)
	st.LineNo = lno
	st.Comp = "Condition";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Kinputp = -1
	st.Kparentp = len( act.ApControlFlow ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Condition has no ControlFlow parent\n") ;
		return 1
	}
	st.Parent = act.ApControlFlow[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Condition under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApControlFlow[ len( act.ApControlFlow )-1 ].Childs = append(act.ApControlFlow[ len( act.ApControlFlow )-1 ].Childs, st)
	act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsCondition = append(act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsCondition, st)	// net5.unit:664, g_structh.act:403
	name,_ := st.Names["condition"]
	s := strconv.Itoa(st.Kparentp) + "_Condition_" + name	// net5.unit:678, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCondition = append(act.ApCondition, st)
	return 0
}

func (me KpCondition) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:680, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "input" { // net5.unit:681, g_structh.act:609
		if (me.Kinputp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kinputp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:664, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:673, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCondition[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Condition > net5.unit:673, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Condition > net5.unit:673, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCondition) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:664, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kparentp ]
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
	if va[0] == "input" {
		if me.Kinputp >= 0 {
			st := glob.Dats.ApTensor[ me.Kinputp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "ProgramInstruction_condition") { // net5.unit:887, g_structh.act:583
		for _, st := range glob.Dats.ApProgramInstruction {
			if (st.Kconditionp == me.Me) {
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
	if (va[0] == "ProgInstruction_condition") { // net5.unit:1033, g_structh.act:583
		for _, st := range glob.Dats.ApProgInstruction {
			if (st.Kconditionp == me.Me) {
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
	        fmt.Printf("?No its %s for Condition %s,%s > net5.unit:673, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBranch struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Klayersp int
}

func (me KpBranch) TypeName() string {
    return me.Comp
}
func (me KpBranch) GetLineNo() string {
	return me.LineNo
}

func loadBranch(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpBranch)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBranch)
	st.LineNo = lno
	st.Comp = "Branch";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Klayersp = -1
	st.Kparentp = len( act.ApControlFlow ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Branch has no ControlFlow parent\n") ;
		return 1
	}
	st.Parent = act.ApControlFlow[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Branch under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApControlFlow[ len( act.ApControlFlow )-1 ].Childs = append(act.ApControlFlow[ len( act.ApControlFlow )-1 ].Childs, st)
	act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsBranch = append(act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsBranch, st)	// net5.unit:664, g_structh.act:403
	name,_ := st.Names["branch"]
	s := strconv.Itoa(st.Kparentp) + "_Branch_" + name	// net5.unit:690, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBranch = append(act.ApBranch, st)
	return 0
}

func (me KpBranch) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:692, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "layers" { // net5.unit:693, g_structh.act:609
		if (me.Klayersp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayersp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:664, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:685, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBranch[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Branch > net5.unit:685, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Branch > net5.unit:685, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBranch) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:664, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kparentp ]
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
	if va[0] == "layers" {
		if me.Klayersp >= 0 {
			st := glob.Dats.ApLayer[ me.Klayersp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Branch %s,%s > net5.unit:685, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpGraphTensor struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kdtypep int
}

func (me KpGraphTensor) TypeName() string {
    return me.Comp
}
func (me KpGraphTensor) GetLineNo() string {
	return me.LineNo
}

func loadGraphTensor(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpGraphTensor)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApGraphTensor)
	st.LineNo = lno
	st.Comp = "GraphTensor";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kdtypep = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " GraphTensor has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " GraphTensor under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsGraphTensor = append(act.ApModel[ len( act.ApModel )-1 ].ItsGraphTensor, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["graph_tensor"]
	s := strconv.Itoa(st.Kparentp) + "_GraphTensor_" + name	// net5.unit:705, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApGraphTensor = append(act.ApGraphTensor, st)
	return 0
}

func (me KpGraphTensor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "dtype" { // net5.unit:711, g_structh.act:609
		if (me.Kdtypep >= 0 && len(va) > 1) {
			return( glob.Dats.ApDtypeRule[ me.Kdtypep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:700, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApGraphTensor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,GraphTensor > net5.unit:700, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,GraphTensor > net5.unit:700, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpGraphTensor) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "dtype" {
		if me.Kdtypep >= 0 {
			st := glob.Dats.ApDtypeRule[ me.Kdtypep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "DynamicGraphNetwork_node_features") { // net5.unit:983, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicGraphNetwork {
			if (st.Knode_featuresp == me.Me) {
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
	if (va[0] == "GraphLearner_input_nodes") { // net5.unit:998, g_structh.act:583
		for _, st := range glob.Dats.ApGraphLearner {
			if (st.Kinput_nodesp == me.Me) {
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
	if (va[0] == "GraphLearner_adjacency") { // net5.unit:1000, g_structh.act:583
		for _, st := range glob.Dats.ApGraphLearner {
			if (st.Kadjacencyp == me.Me) {
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
	        fmt.Printf("?No its %s for GraphTensor %s,%s > net5.unit:700, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpStateTransfer struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Ksource_statep int
	Kmodel2p int
	Ktarget_statep int
}

func (me KpStateTransfer) TypeName() string {
    return me.Comp
}
func (me KpStateTransfer) GetLineNo() string {
	return me.LineNo
}

func loadStateTransfer(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpStateTransfer)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApStateTransfer)
	st.LineNo = lno
	st.Comp = "StateTransfer";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Ksource_statep = -1
	st.Kmodel2p = -1
	st.Ktarget_statep = -1
	st.Kparentp = len( act.ApConfig ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " StateTransfer has no Config parent\n") ;
		return 1
	}
	st.Parent = act.ApConfig[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " StateTransfer under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApConfig[ len( act.ApConfig )-1 ].Childs = append(act.ApConfig[ len( act.ApConfig )-1 ].Childs, st)
	act.ApConfig[ len( act.ApConfig )-1 ].ItsStateTransfer = append(act.ApConfig[ len( act.ApConfig )-1 ].ItsStateTransfer, st)	// net5.unit:633, g_structh.act:403
	name,_ := st.Names["transfer"]
	s := strconv.Itoa(st.Kparentp) + "_StateTransfer_" + name	// net5.unit:720, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApStateTransfer = append(act.ApStateTransfer, st)
	return 0
}

func (me KpStateTransfer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:722, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "source_state" { // net5.unit:723, g_structh.act:609
		if (me.Ksource_statep >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ksource_statep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:724, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "target_state" { // net5.unit:725, g_structh.act:609
		if (me.Ktarget_statep >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktarget_statep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:633, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:715, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApStateTransfer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,StateTransfer > net5.unit:715, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,StateTransfer > net5.unit:715, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpStateTransfer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:633, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApConfig[ me.Kparentp ]
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
	if va[0] == "source_state" {
		if me.Ksource_statep >= 0 {
			st := glob.Dats.ApTensor[ me.Ksource_statep ]
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
	if va[0] == "target_state" {
		if me.Ktarget_statep >= 0 {
			st := glob.Dats.ApTensor[ me.Ktarget_statep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for StateTransfer %s,%s > net5.unit:715, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpContinuousLayer struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Kdynamicsp int
}

func (me KpContinuousLayer) TypeName() string {
    return me.Comp
}
func (me KpContinuousLayer) GetLineNo() string {
	return me.LineNo
}

func loadContinuousLayer(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpContinuousLayer)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApContinuousLayer)
	st.LineNo = lno
	st.Comp = "ContinuousLayer";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Kdynamicsp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ContinuousLayer has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ContinuousLayer under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsContinuousLayer = append(act.ApModel[ len( act.ApModel )-1 ].ItsContinuousLayer, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["cont_layer"]
	s := strconv.Itoa(st.Kparentp) + "_ContinuousLayer_" + name	// net5.unit:737, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApContinuousLayer = append(act.ApContinuousLayer, st)
	return 0
}

func (me KpContinuousLayer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:743, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "dynamics" { // net5.unit:744, g_structh.act:609
		if (me.Kdynamicsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kdynamicsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:732, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApContinuousLayer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ContinuousLayer > net5.unit:732, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ContinuousLayer > net5.unit:732, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpContinuousLayer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "dynamics" {
		if me.Kdynamicsp >= 0 {
			st := glob.Dats.ApLayer[ me.Kdynamicsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ContinuousLayer %s,%s > net5.unit:732, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
	ItsSearchOp [] *KpSearchOp 
	ItsSearchEdge [] *KpSearchEdge 
	Childs [] Kp
}

func (me KpSearchSpace) TypeName() string {
    return me.Comp
}
func (me KpSearchSpace) GetLineNo() string {
	return me.LineNo
}

func loadSearchSpace(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchSpace has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchSpace under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsSearchSpace = append(act.ApModel[ len( act.ApModel )-1 ].ItsSearchSpace, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["search_space"]
	s := strconv.Itoa(st.Kparentp) + "_SearchSpace_" + name	// net5.unit:758, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchSpace = append(act.ApSearchSpace, st)
	return 0
}

func (me KpSearchSpace) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:753, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpace > net5.unit:753, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchSpace > net5.unit:753, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchSpace) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "SearchOp" { // net5.unit:763, g_structh.act:676
		for _, st := range me.ItsSearchOp {
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
	if va[0] == "SearchEdge" { // net5.unit:776, g_structh.act:676
		for _, st := range me.ItsSearchEdge {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchSpace %s,%s > net5.unit:753, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Klayerp int
	Koperationp int
}

func (me KpSearchOp) TypeName() string {
    return me.Comp
}
func (me KpSearchOp) GetLineNo() string {
	return me.LineNo
}

func loadSearchOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchOp)
	st.LineNo = lno
	st.Comp = "SearchOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Klayerp = -1
	st.Koperationp = -1
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchOp has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchOp = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchOp, st)	// net5.unit:753, g_structh.act:403
	name,_ := st.Names["search_op"]
	s := strconv.Itoa(st.Kparentp) + "_SearchOp_" + name	// net5.unit:768, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchOp = append(act.ApSearchOp, st)
	return 0
}

func (me KpSearchOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:769, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "layer" { // net5.unit:770, g_structh.act:609
		if (me.Klayerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "operation" { // net5.unit:771, g_structh.act:609
		if (me.Koperationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Koperationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:753, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:763, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchOp > net5.unit:763, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchOp > net5.unit:763, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:753, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
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
	if va[0] == "operation" {
		if me.Koperationp >= 0 {
			st := glob.Dats.ApOp[ me.Koperationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchOp %s,%s > net5.unit:763, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchEdge struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpSearchEdge) TypeName() string {
    return me.Comp
}
func (me KpSearchEdge) GetLineNo() string {
	return me.LineNo
}

func loadSearchEdge(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchEdge)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchEdge)
	st.LineNo = lno
	st.Comp = "SearchEdge";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchEdge has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchEdge under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchEdge = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchEdge, st)	// net5.unit:753, g_structh.act:403
	name,_ := st.Names["search_edge"]
	s := strconv.Itoa(st.Kparentp) + "_SearchEdge_" + name	// net5.unit:781, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchEdge = append(act.ApSearchEdge, st)
	return 0
}

func (me KpSearchEdge) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:753, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:776, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchEdge[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchEdge > net5.unit:776, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchEdge > net5.unit:776, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchEdge) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:753, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchEdge %s,%s > net5.unit:776, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpArchitectureParam struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpArchitectureParam) TypeName() string {
    return me.Comp
}
func (me KpArchitectureParam) GetLineNo() string {
	return me.LineNo
}

func loadArchitectureParam(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpArchitectureParam)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApArchitectureParam)
	st.LineNo = lno
	st.Comp = "ArchitectureParam";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ArchitectureParam has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ArchitectureParam under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsArchitectureParam = append(act.ApModel[ len( act.ApModel )-1 ].ItsArchitectureParam, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["arch_param"]
	s := strconv.Itoa(st.Kparentp) + "_ArchitectureParam_" + name	// net5.unit:792, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArchitectureParam = append(act.ApArchitectureParam, st)
	return 0
}

func (me KpArchitectureParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:787, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArchitectureParam > net5.unit:787, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArchitectureParam > net5.unit:787, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitectureParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "ArchitectureGradient_arch_params") { // net5.unit:805, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureGradient {
			if (st.Karch_paramsp == me.Me) {
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
	        fmt.Printf("?No its %s for ArchitectureParam %s,%s > net5.unit:787, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpArchitectureGradient struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Karch_paramsp int
	Kmodel2p int
	Klayerp int
	Kgradient_opsp int
}

func (me KpArchitectureGradient) TypeName() string {
    return me.Comp
}
func (me KpArchitectureGradient) GetLineNo() string {
	return me.LineNo
}

func loadArchitectureGradient(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpArchitectureGradient)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApArchitectureGradient)
	st.LineNo = lno
	st.Comp = "ArchitectureGradient";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Karch_paramsp = -1
	st.Kmodel2p = -1
	st.Klayerp = -1
	st.Kgradient_opsp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ArchitectureGradient has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ArchitectureGradient under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsArchitectureGradient = append(act.ApModel[ len( act.ApModel )-1 ].ItsArchitectureGradient, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["arch_gradient"]
	s := strconv.Itoa(st.Kparentp) + "_ArchitectureGradient_" + name	// net5.unit:803, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArchitectureGradient = append(act.ApArchitectureGradient, st)
	return 0
}

func (me KpArchitectureGradient) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:804, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "arch_params" { // net5.unit:805, g_structh.act:609
		if (me.Karch_paramsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureParam[ me.Karch_paramsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:806, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "layer" { // net5.unit:807, g_structh.act:609
		if (me.Klayerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "gradient_ops" { // net5.unit:808, g_structh.act:609
		if (me.Kgradient_opsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kgradient_opsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:798, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureGradient[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArchitectureGradient > net5.unit:798, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArchitectureGradient > net5.unit:798, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitectureGradient) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "arch_params" {
		if me.Karch_paramsp >= 0 {
			st := glob.Dats.ApArchitectureParam[ me.Karch_paramsp ]
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
	if va[0] == "gradient_ops" {
		if me.Kgradient_opsp >= 0 {
			st := glob.Dats.ApOp[ me.Kgradient_opsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ArchitectureGradient %s,%s > net5.unit:798, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHyperNetwork struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Ktarget_netp int
	Kmodel2p int
	Kconditionp int
	ItsWeightGenerator [] *KpWeightGenerator 
	Childs [] Kp
}

func (me KpHyperNetwork) TypeName() string {
    return me.Comp
}
func (me KpHyperNetwork) GetLineNo() string {
	return me.LineNo
}

func loadHyperNetwork(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpHyperNetwork)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHyperNetwork)
	st.LineNo = lno
	st.Comp = "HyperNetwork";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Ktarget_netp = -1
	st.Kmodel2p = -1
	st.Kconditionp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " HyperNetwork has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " HyperNetwork under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsHyperNetwork = append(act.ApModel[ len( act.ApModel )-1 ].ItsHyperNetwork, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["hypernet"]
	s := strconv.Itoa(st.Kparentp) + "_HyperNetwork_" + name	// net5.unit:822, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApHyperNetwork = append(act.ApHyperNetwork, st)
	return 0
}

func (me KpHyperNetwork) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:823, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "target_net" { // net5.unit:824, g_structh.act:609
		if (me.Ktarget_netp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Ktarget_netp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:827, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "condition" { // net5.unit:828, g_structh.act:609
		if (me.Kconditionp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kconditionp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:817, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHyperNetwork[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,HyperNetwork > net5.unit:817, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,HyperNetwork > net5.unit:817, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHyperNetwork) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "WeightGenerator" { // net5.unit:831, g_structh.act:676
		for _, st := range me.ItsWeightGenerator {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "target_net" {
		if me.Ktarget_netp >= 0 {
			st := glob.Dats.ApLayer[ me.Ktarget_netp ]
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
	if va[0] == "condition" {
		if me.Kconditionp >= 0 {
			st := glob.Dats.ApTensor[ me.Kconditionp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for HyperNetwork %s,%s > net5.unit:817, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpWeightGenerator struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Kinput_tensorp int
	Kmodel2p int
	Koutput_tensorp int
}

func (me KpWeightGenerator) TypeName() string {
    return me.Comp
}
func (me KpWeightGenerator) GetLineNo() string {
	return me.LineNo
}

func loadWeightGenerator(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpWeightGenerator)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApWeightGenerator)
	st.LineNo = lno
	st.Comp = "WeightGenerator";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kinput_tensorp = -1
	st.Kmodel2p = -1
	st.Koutput_tensorp = -1
	st.Kparentp = len( act.ApHyperNetwork ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " WeightGenerator has no HyperNetwork parent\n") ;
		return 1
	}
	st.Parent = act.ApHyperNetwork[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " WeightGenerator under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHyperNetwork[ len( act.ApHyperNetwork )-1 ].Childs = append(act.ApHyperNetwork[ len( act.ApHyperNetwork )-1 ].Childs, st)
	act.ApHyperNetwork[ len( act.ApHyperNetwork )-1 ].ItsWeightGenerator = append(act.ApHyperNetwork[ len( act.ApHyperNetwork )-1 ].ItsWeightGenerator, st)	// net5.unit:817, g_structh.act:403
	name,_ := st.Names["generator"]
	s := strconv.Itoa(st.Kparentp) + "_WeightGenerator_" + name	// net5.unit:836, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApWeightGenerator = append(act.ApWeightGenerator, st)
	return 0
}

func (me KpWeightGenerator) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:838, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "input_tensor" { // net5.unit:839, g_structh.act:609
		if (me.Kinput_tensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kinput_tensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:840, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "output_tensor" { // net5.unit:841, g_structh.act:609
		if (me.Koutput_tensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Koutput_tensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:817, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHyperNetwork[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:831, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApWeightGenerator[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,WeightGenerator > net5.unit:831, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,WeightGenerator > net5.unit:831, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpWeightGenerator) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:817, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHyperNetwork[ me.Kparentp ]
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
	if va[0] == "input_tensor" {
		if me.Kinput_tensorp >= 0 {
			st := glob.Dats.ApTensor[ me.Kinput_tensorp ]
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
	if va[0] == "output_tensor" {
		if me.Koutput_tensorp >= 0 {
			st := glob.Dats.ApTensor[ me.Koutput_tensorp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for WeightGenerator %s,%s > net5.unit:831, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMemoryMatrix struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpMemoryMatrix) TypeName() string {
    return me.Comp
}
func (me KpMemoryMatrix) GetLineNo() string {
	return me.LineNo
}

func loadMemoryMatrix(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMemoryMatrix)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMemoryMatrix)
	st.LineNo = lno
	st.Comp = "MemoryMatrix";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MemoryMatrix has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MemoryMatrix under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsMemoryMatrix = append(act.ApModel[ len( act.ApModel )-1 ].ItsMemoryMatrix, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["memory"]
	s := strconv.Itoa(st.Kparentp) + "_MemoryMatrix_" + name	// net5.unit:853, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApMemoryMatrix = append(act.ApMemoryMatrix, st)
	return 0
}

func (me KpMemoryMatrix) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:848, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMemoryMatrix[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MemoryMatrix > net5.unit:848, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MemoryMatrix > net5.unit:848, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMemoryMatrix) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MemoryMatrix %s,%s > net5.unit:848, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDifferentiableProgram struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsProgramInstruction [] *KpProgramInstruction 
	Childs [] Kp
}

func (me KpDifferentiableProgram) TypeName() string {
    return me.Comp
}
func (me KpDifferentiableProgram) GetLineNo() string {
	return me.LineNo
}

func loadDifferentiableProgram(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDifferentiableProgram)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDifferentiableProgram)
	st.LineNo = lno
	st.Comp = "DifferentiableProgram";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DifferentiableProgram has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DifferentiableProgram under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsDifferentiableProgram = append(act.ApModel[ len( act.ApModel )-1 ].ItsDifferentiableProgram, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["program"]
	s := strconv.Itoa(st.Kparentp) + "_DifferentiableProgram_" + name	// net5.unit:870, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDifferentiableProgram = append(act.ApDifferentiableProgram, st)
	return 0
}

func (me KpDifferentiableProgram) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:865, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDifferentiableProgram[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DifferentiableProgram > net5.unit:865, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DifferentiableProgram > net5.unit:865, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDifferentiableProgram) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ProgramInstruction" { // net5.unit:876, g_structh.act:676
		for _, st := range me.ItsProgramInstruction {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DifferentiableProgram %s,%s > net5.unit:865, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProgramInstruction struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Koperandsp int
	Kmodel2p int
	Kcontrol_flowp int
	Kconditionp int
}

func (me KpProgramInstruction) TypeName() string {
    return me.Comp
}
func (me KpProgramInstruction) GetLineNo() string {
	return me.LineNo
}

func loadProgramInstruction(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpProgramInstruction)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProgramInstruction)
	st.LineNo = lno
	st.Comp = "ProgramInstruction";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Koperandsp = -1
	st.Kmodel2p = -1
	st.Kcontrol_flowp = -1
	st.Kconditionp = -1
	st.Kparentp = len( act.ApDifferentiableProgram ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ProgramInstruction has no DifferentiableProgram parent\n") ;
		return 1
	}
	st.Parent = act.ApDifferentiableProgram[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ProgramInstruction under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApDifferentiableProgram[ len( act.ApDifferentiableProgram )-1 ].Childs = append(act.ApDifferentiableProgram[ len( act.ApDifferentiableProgram )-1 ].Childs, st)
	act.ApDifferentiableProgram[ len( act.ApDifferentiableProgram )-1 ].ItsProgramInstruction = append(act.ApDifferentiableProgram[ len( act.ApDifferentiableProgram )-1 ].ItsProgramInstruction, st)	// net5.unit:865, g_structh.act:403
	name,_ := st.Names["instruction"]
	s := strconv.Itoa(st.Kparentp) + "_ProgramInstruction_" + name	// net5.unit:881, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApProgramInstruction = append(act.ApProgramInstruction, st)
	return 0
}

func (me KpProgramInstruction) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:883, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "operands" { // net5.unit:884, g_structh.act:609
		if (me.Koperandsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Koperandsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:885, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "control_flow" { // net5.unit:886, g_structh.act:609
		if (me.Kcontrol_flowp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kcontrol_flowp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "condition" { // net5.unit:887, g_structh.act:609
		if (me.Kconditionp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCondition[ me.Kconditionp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:865, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDifferentiableProgram[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:876, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProgramInstruction[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ProgramInstruction > net5.unit:876, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ProgramInstruction > net5.unit:876, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProgramInstruction) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:865, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApDifferentiableProgram[ me.Kparentp ]
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
	if va[0] == "operands" {
		if me.Koperandsp >= 0 {
			st := glob.Dats.ApTensor[ me.Koperandsp ]
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
	if va[0] == "control_flow" {
		if me.Kcontrol_flowp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kcontrol_flowp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "condition" {
		if me.Kconditionp >= 0 {
			st := glob.Dats.ApCondition[ me.Kconditionp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ProgramInstruction %s,%s > net5.unit:876, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMetaLearner struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Kinner_loopp int
	Kmodel2p int
	Kouter_loopp int
}

func (me KpMetaLearner) TypeName() string {
    return me.Comp
}
func (me KpMetaLearner) GetLineNo() string {
	return me.LineNo
}

func loadMetaLearner(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMetaLearner)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMetaLearner)
	st.LineNo = lno
	st.Comp = "MetaLearner";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kinner_loopp = -1
	st.Kmodel2p = -1
	st.Kouter_loopp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MetaLearner has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MetaLearner under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsMetaLearner = append(act.ApModel[ len( act.ApModel )-1 ].ItsMetaLearner, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["meta_learner"]
	s := strconv.Itoa(st.Kparentp) + "_MetaLearner_" + name	// net5.unit:896, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApMetaLearner = append(act.ApMetaLearner, st)
	return 0
}

func (me KpMetaLearner) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:897, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "inner_loop" { // net5.unit:898, g_structh.act:609
		if (me.Kinner_loopp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kinner_loopp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:899, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "outer_loop" { // net5.unit:900, g_structh.act:609
		if (me.Kouter_loopp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kouter_loopp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:891, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMetaLearner[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MetaLearner > net5.unit:891, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MetaLearner > net5.unit:891, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMetaLearner) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "inner_loop" {
		if me.Kinner_loopp >= 0 {
			st := glob.Dats.ApLayer[ me.Kinner_loopp ]
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
	if va[0] == "outer_loop" {
		if me.Kouter_loopp >= 0 {
			st := glob.Dats.ApLayer[ me.Kouter_loopp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MetaLearner %s,%s > net5.unit:891, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSparseExpertSystem struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Kexpertsp int
	ItsCapacityAwareRouter [] *KpCapacityAwareRouter 
	Childs [] Kp
}

func (me KpSparseExpertSystem) TypeName() string {
    return me.Comp
}
func (me KpSparseExpertSystem) GetLineNo() string {
	return me.LineNo
}

func loadSparseExpertSystem(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSparseExpertSystem)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSparseExpertSystem)
	st.LineNo = lno
	st.Comp = "SparseExpertSystem";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Kexpertsp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SparseExpertSystem has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SparseExpertSystem under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsSparseExpertSystem = append(act.ApModel[ len( act.ApModel )-1 ].ItsSparseExpertSystem, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["expert_system"]
	s := strconv.Itoa(st.Kparentp) + "_SparseExpertSystem_" + name	// net5.unit:914, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSparseExpertSystem = append(act.ApSparseExpertSystem, st)
	return 0
}

func (me KpSparseExpertSystem) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:915, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "experts" { // net5.unit:916, g_structh.act:609
		if (me.Kexpertsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kexpertsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:909, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSparseExpertSystem[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SparseExpertSystem > net5.unit:909, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SparseExpertSystem > net5.unit:909, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSparseExpertSystem) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "CapacityAwareRouter" { // net5.unit:923, g_structh.act:676
		for _, st := range me.ItsCapacityAwareRouter {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "experts" {
		if me.Kexpertsp >= 0 {
			st := glob.Dats.ApLayer[ me.Kexpertsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SparseExpertSystem %s,%s > net5.unit:909, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCapacityAwareRouter struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Kgate_inputp int
	Kmodel2p int
	Kexpert_maskp int
	Kmodel3p int
	Ktoken_assignp int
}

func (me KpCapacityAwareRouter) TypeName() string {
    return me.Comp
}
func (me KpCapacityAwareRouter) GetLineNo() string {
	return me.LineNo
}

func loadCapacityAwareRouter(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCapacityAwareRouter)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCapacityAwareRouter)
	st.LineNo = lno
	st.Comp = "CapacityAwareRouter";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kgate_inputp = -1
	st.Kmodel2p = -1
	st.Kexpert_maskp = -1
	st.Kmodel3p = -1
	st.Ktoken_assignp = -1
	st.Kparentp = len( act.ApSparseExpertSystem ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " CapacityAwareRouter has no SparseExpertSystem parent\n") ;
		return 1
	}
	st.Parent = act.ApSparseExpertSystem[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " CapacityAwareRouter under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSparseExpertSystem[ len( act.ApSparseExpertSystem )-1 ].Childs = append(act.ApSparseExpertSystem[ len( act.ApSparseExpertSystem )-1 ].Childs, st)
	act.ApSparseExpertSystem[ len( act.ApSparseExpertSystem )-1 ].ItsCapacityAwareRouter = append(act.ApSparseExpertSystem[ len( act.ApSparseExpertSystem )-1 ].ItsCapacityAwareRouter, st)	// net5.unit:909, g_structh.act:403
	name,_ := st.Names["router"]
	s := strconv.Itoa(st.Kparentp) + "_CapacityAwareRouter_" + name	// net5.unit:928, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCapacityAwareRouter = append(act.ApCapacityAwareRouter, st)
	return 0
}

func (me KpCapacityAwareRouter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:929, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "gate_input" { // net5.unit:930, g_structh.act:609
		if (me.Kgate_inputp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kgate_inputp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:931, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "expert_mask" { // net5.unit:932, g_structh.act:609
		if (me.Kexpert_maskp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kexpert_maskp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model3" { // net5.unit:933, g_structh.act:609
		if (me.Kmodel3p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel3p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "token_assign" { // net5.unit:934, g_structh.act:609
		if (me.Ktoken_assignp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktoken_assignp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:909, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSparseExpertSystem[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:923, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCapacityAwareRouter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CapacityAwareRouter > net5.unit:923, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CapacityAwareRouter > net5.unit:923, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCapacityAwareRouter) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:909, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSparseExpertSystem[ me.Kparentp ]
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
	if va[0] == "gate_input" {
		if me.Kgate_inputp >= 0 {
			st := glob.Dats.ApTensor[ me.Kgate_inputp ]
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
	if va[0] == "expert_mask" {
		if me.Kexpert_maskp >= 0 {
			st := glob.Dats.ApTensor[ me.Kexpert_maskp ]
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
	if va[0] == "token_assign" {
		if me.Ktoken_assignp >= 0 {
			st := glob.Dats.ApTensor[ me.Ktoken_assignp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CapacityAwareRouter %s,%s > net5.unit:923, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuralSDE struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Kdrift_netp int
	Kmodel2p int
	Kdiffusion_netp int
}

func (me KpNeuralSDE) TypeName() string {
    return me.Comp
}
func (me KpNeuralSDE) GetLineNo() string {
	return me.LineNo
}

func loadNeuralSDE(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuralSDE)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuralSDE)
	st.LineNo = lno
	st.Comp = "NeuralSDE";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kdrift_netp = -1
	st.Kmodel2p = -1
	st.Kdiffusion_netp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralSDE has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralSDE under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsNeuralSDE = append(act.ApModel[ len( act.ApModel )-1 ].ItsNeuralSDE, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["neural_sde"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralSDE_" + name	// net5.unit:947, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralSDE = append(act.ApNeuralSDE, st)
	return 0
}

func (me KpNeuralSDE) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:948, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "drift_net" { // net5.unit:949, g_structh.act:609
		if (me.Kdrift_netp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kdrift_netp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:950, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "diffusion_net" { // net5.unit:951, g_structh.act:609
		if (me.Kdiffusion_netp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kdiffusion_netp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:942, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralSDE[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralSDE > net5.unit:942, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralSDE > net5.unit:942, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralSDE) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "drift_net" {
		if me.Kdrift_netp >= 0 {
			st := glob.Dats.ApLayer[ me.Kdrift_netp ]
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
	if va[0] == "diffusion_net" {
		if me.Kdiffusion_netp >= 0 {
			st := glob.Dats.ApLayer[ me.Kdiffusion_netp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralSDE %s,%s > net5.unit:942, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuralPDE struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Kdomainp int
	Kmodel2p int
	Kboundaryp int
	Kmodel3p int
	Kinitialp int
}

func (me KpNeuralPDE) TypeName() string {
    return me.Comp
}
func (me KpNeuralPDE) GetLineNo() string {
	return me.LineNo
}

func loadNeuralPDE(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuralPDE)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuralPDE)
	st.LineNo = lno
	st.Comp = "NeuralPDE";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kdomainp = -1
	st.Kmodel2p = -1
	st.Kboundaryp = -1
	st.Kmodel3p = -1
	st.Kinitialp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralPDE has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralPDE under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsNeuralPDE = append(act.ApModel[ len( act.ApModel )-1 ].ItsNeuralPDE, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["neural_pde"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralPDE_" + name	// net5.unit:961, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralPDE = append(act.ApNeuralPDE, st)
	return 0
}

func (me KpNeuralPDE) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:963, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "domain" { // net5.unit:964, g_structh.act:609
		if (me.Kdomainp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kdomainp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:965, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "boundary" { // net5.unit:966, g_structh.act:609
		if (me.Kboundaryp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kboundaryp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model3" { // net5.unit:967, g_structh.act:609
		if (me.Kmodel3p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel3p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "initial" { // net5.unit:968, g_structh.act:609
		if (me.Kinitialp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kinitialp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:956, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralPDE[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralPDE > net5.unit:956, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralPDE > net5.unit:956, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralPDE) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "domain" {
		if me.Kdomainp >= 0 {
			st := glob.Dats.ApTensor[ me.Kdomainp ]
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
	if va[0] == "boundary" {
		if me.Kboundaryp >= 0 {
			st := glob.Dats.ApTensor[ me.Kboundaryp ]
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
	if va[0] == "initial" {
		if me.Kinitialp >= 0 {
			st := glob.Dats.ApTensor[ me.Kinitialp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralPDE %s,%s > net5.unit:956, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDynamicGraphNetwork struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Knode_featuresp int
	Kmodel2p int
	Kedge_predictorp int
	ItsGraphLearner [] *KpGraphLearner 
	Childs [] Kp
}

func (me KpDynamicGraphNetwork) TypeName() string {
    return me.Comp
}
func (me KpDynamicGraphNetwork) GetLineNo() string {
	return me.LineNo
}

func loadDynamicGraphNetwork(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDynamicGraphNetwork)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDynamicGraphNetwork)
	st.LineNo = lno
	st.Comp = "DynamicGraphNetwork";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Knode_featuresp = -1
	st.Kmodel2p = -1
	st.Kedge_predictorp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DynamicGraphNetwork has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DynamicGraphNetwork under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsDynamicGraphNetwork = append(act.ApModel[ len( act.ApModel )-1 ].ItsDynamicGraphNetwork, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["dynamic_graph"]
	s := strconv.Itoa(st.Kparentp) + "_DynamicGraphNetwork_" + name	// net5.unit:981, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDynamicGraphNetwork = append(act.ApDynamicGraphNetwork, st)
	return 0
}

func (me KpDynamicGraphNetwork) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:982, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "node_features" { // net5.unit:983, g_structh.act:609
		if (me.Knode_featuresp >= 0 && len(va) > 1) {
			return( glob.Dats.ApGraphTensor[ me.Knode_featuresp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:984, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "edge_predictor" { // net5.unit:985, g_structh.act:609
		if (me.Kedge_predictorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kedge_predictorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:976, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicGraphNetwork[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DynamicGraphNetwork > net5.unit:976, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DynamicGraphNetwork > net5.unit:976, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDynamicGraphNetwork) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "GraphLearner" { // net5.unit:991, g_structh.act:676
		for _, st := range me.ItsGraphLearner {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "node_features" {
		if me.Knode_featuresp >= 0 {
			st := glob.Dats.ApGraphTensor[ me.Knode_featuresp ]
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
	if va[0] == "edge_predictor" {
		if me.Kedge_predictorp >= 0 {
			st := glob.Dats.ApLayer[ me.Kedge_predictorp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DynamicGraphNetwork %s,%s > net5.unit:976, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpGraphLearner struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Kinput_nodesp int
	Kmodel2p int
	Kadjacencyp int
}

func (me KpGraphLearner) TypeName() string {
    return me.Comp
}
func (me KpGraphLearner) GetLineNo() string {
	return me.LineNo
}

func loadGraphLearner(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpGraphLearner)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApGraphLearner)
	st.LineNo = lno
	st.Comp = "GraphLearner";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Kinput_nodesp = -1
	st.Kmodel2p = -1
	st.Kadjacencyp = -1
	st.Kparentp = len( act.ApDynamicGraphNetwork ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " GraphLearner has no DynamicGraphNetwork parent\n") ;
		return 1
	}
	st.Parent = act.ApDynamicGraphNetwork[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " GraphLearner under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApDynamicGraphNetwork[ len( act.ApDynamicGraphNetwork )-1 ].Childs = append(act.ApDynamicGraphNetwork[ len( act.ApDynamicGraphNetwork )-1 ].Childs, st)
	act.ApDynamicGraphNetwork[ len( act.ApDynamicGraphNetwork )-1 ].ItsGraphLearner = append(act.ApDynamicGraphNetwork[ len( act.ApDynamicGraphNetwork )-1 ].ItsGraphLearner, st)	// net5.unit:976, g_structh.act:403
	name,_ := st.Names["learner"]
	s := strconv.Itoa(st.Kparentp) + "_GraphLearner_" + name	// net5.unit:996, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApGraphLearner = append(act.ApGraphLearner, st)
	return 0
}

func (me KpGraphLearner) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:997, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "input_nodes" { // net5.unit:998, g_structh.act:609
		if (me.Kinput_nodesp >= 0 && len(va) > 1) {
			return( glob.Dats.ApGraphTensor[ me.Kinput_nodesp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:999, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "adjacency" { // net5.unit:1000, g_structh.act:609
		if (me.Kadjacencyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApGraphTensor[ me.Kadjacencyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:976, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicGraphNetwork[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:991, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApGraphLearner[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,GraphLearner > net5.unit:991, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,GraphLearner > net5.unit:991, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpGraphLearner) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:976, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApDynamicGraphNetwork[ me.Kparentp ]
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
	if va[0] == "input_nodes" {
		if me.Kinput_nodesp >= 0 {
			st := glob.Dats.ApGraphTensor[ me.Kinput_nodesp ]
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
	if va[0] == "adjacency" {
		if me.Kadjacencyp >= 0 {
			st := glob.Dats.ApGraphTensor[ me.Kadjacencyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for GraphLearner %s,%s > net5.unit:991, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuralProgram struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Kmemoryp int
	ItsProgInstruction [] *KpProgInstruction 
	Childs [] Kp
}

func (me KpNeuralProgram) TypeName() string {
    return me.Comp
}
func (me KpNeuralProgram) GetLineNo() string {
	return me.LineNo
}

func loadNeuralProgram(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuralProgram)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuralProgram)
	st.LineNo = lno
	st.Comp = "NeuralProgram";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Kmemoryp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralProgram has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralProgram under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsNeuralProgram = append(act.ApModel[ len( act.ApModel )-1 ].ItsNeuralProgram, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["neural_prog"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralProgram_" + name	// net5.unit:1014, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralProgram = append(act.ApNeuralProgram, st)
	return 0
}

func (me KpNeuralProgram) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:1017, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "memory" { // net5.unit:1018, g_structh.act:609
		if (me.Kmemoryp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kmemoryp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:1009, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralProgram[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralProgram > net5.unit:1009, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralProgram > net5.unit:1009, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralProgram) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ProgInstruction" { // net5.unit:1022, g_structh.act:676
		for _, st := range me.ItsProgInstruction {
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
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "memory" {
		if me.Kmemoryp >= 0 {
			st := glob.Dats.ApTensor[ me.Kmemoryp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralProgram %s,%s > net5.unit:1009, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProgInstruction struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodel1p int
	Koperandsp int
	Kmodel2p int
	Kcontrol_flowp int
	Kconditionp int
}

func (me KpProgInstruction) TypeName() string {
    return me.Comp
}
func (me KpProgInstruction) GetLineNo() string {
	return me.LineNo
}

func loadProgInstruction(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpProgInstruction)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProgInstruction)
	st.LineNo = lno
	st.Comp = "ProgInstruction";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodel1p = -1
	st.Koperandsp = -1
	st.Kmodel2p = -1
	st.Kcontrol_flowp = -1
	st.Kconditionp = -1
	st.Kparentp = len( act.ApNeuralProgram ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ProgInstruction has no NeuralProgram parent\n") ;
		return 1
	}
	st.Parent = act.ApNeuralProgram[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ProgInstruction under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApNeuralProgram[ len( act.ApNeuralProgram )-1 ].Childs = append(act.ApNeuralProgram[ len( act.ApNeuralProgram )-1 ].Childs, st)
	act.ApNeuralProgram[ len( act.ApNeuralProgram )-1 ].ItsProgInstruction = append(act.ApNeuralProgram[ len( act.ApNeuralProgram )-1 ].ItsProgInstruction, st)	// net5.unit:1009, g_structh.act:403
	name,_ := st.Names["prog_inst"]
	s := strconv.Itoa(st.Kparentp) + "_ProgInstruction_" + name	// net5.unit:1027, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApProgInstruction = append(act.ApProgInstruction, st)
	return 0
}

func (me KpProgInstruction) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model1" { // net5.unit:1029, g_structh.act:609
		if (me.Kmodel1p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel1p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "operands" { // net5.unit:1030, g_structh.act:609
		if (me.Koperandsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Koperandsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model2" { // net5.unit:1031, g_structh.act:609
		if (me.Kmodel2p >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodel2p ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "control_flow" { // net5.unit:1032, g_structh.act:609
		if (me.Kcontrol_flowp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kcontrol_flowp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "condition" { // net5.unit:1033, g_structh.act:609
		if (me.Kconditionp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCondition[ me.Kconditionp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:1009, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralProgram[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:1022, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProgInstruction[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ProgInstruction > net5.unit:1022, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ProgInstruction > net5.unit:1022, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProgInstruction) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:1009, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApNeuralProgram[ me.Kparentp ]
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
	if va[0] == "operands" {
		if me.Koperandsp >= 0 {
			st := glob.Dats.ApTensor[ me.Koperandsp ]
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
	if va[0] == "control_flow" {
		if me.Kcontrol_flowp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kcontrol_flowp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "condition" {
		if me.Kconditionp >= 0 {
			st := glob.Dats.ApCondition[ me.Kconditionp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ProgInstruction %s,%s > net5.unit:1022, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpLiquidNetwork struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmodelp int
	Ktime_constp int
}

func (me KpLiquidNetwork) TypeName() string {
    return me.Comp
}
func (me KpLiquidNetwork) GetLineNo() string {
	return me.LineNo
}

func loadLiquidNetwork(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpLiquidNetwork)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApLiquidNetwork)
	st.LineNo = lno
	st.Comp = "LiquidNetwork";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmodelp = -1
	st.Ktime_constp = -1
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " LiquidNetwork has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " LiquidNetwork under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsLiquidNetwork = append(act.ApModel[ len( act.ApModel )-1 ].ItsLiquidNetwork, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["liquid_net"]
	s := strconv.Itoa(st.Kparentp) + "_LiquidNetwork_" + name	// net5.unit:1046, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApLiquidNetwork = append(act.ApLiquidNetwork, st)
	return 0
}

func (me KpLiquidNetwork) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "model" { // net5.unit:1047, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "time_const" { // net5.unit:1048, g_structh.act:609
		if (me.Ktime_constp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktime_constp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:1041, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLiquidNetwork[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,LiquidNetwork > net5.unit:1041, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,LiquidNetwork > net5.unit:1041, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpLiquidNetwork) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
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
	if va[0] == "time_const" {
		if me.Ktime_constp >= 0 {
			st := glob.Dats.ApTensor[ me.Ktime_constp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for LiquidNetwork %s,%s > net5.unit:1041, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSymbolicShape struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpSymbolicShape) TypeName() string {
    return me.Comp
}
func (me KpSymbolicShape) GetLineNo() string {
	return me.LineNo
}

func loadSymbolicShape(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSymbolicShape)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSymbolicShape)
	st.LineNo = lno
	st.Comp = "SymbolicShape";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SymbolicShape has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SymbolicShape under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsSymbolicShape = append(act.ApModel[ len( act.ApModel )-1 ].ItsSymbolicShape, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["symbolic_shape"]
	s := strconv.Itoa(st.Kparentp) + "_SymbolicShape_" + name	// net5.unit:1063, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSymbolicShape = append(act.ApSymbolicShape, st)
	return 0
}

func (me KpSymbolicShape) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:1058, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSymbolicShape[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SymbolicShape > net5.unit:1058, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SymbolicShape > net5.unit:1058, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSymbolicShape) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SymbolicShape %s,%s > net5.unit:1058, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpJITCompiler struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
}

func (me KpJITCompiler) TypeName() string {
    return me.Comp
}
func (me KpJITCompiler) GetLineNo() string {
	return me.LineNo
}

func loadJITCompiler(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpJITCompiler)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApJITCompiler)
	st.LineNo = lno
	st.Comp = "JITCompiler";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApModel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " JITCompiler has no Model parent\n") ;
		return 1
	}
	st.Parent = act.ApModel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " JITCompiler under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApModel[ len( act.ApModel )-1 ].Childs = append(act.ApModel[ len( act.ApModel )-1 ].Childs, st)
	act.ApModel[ len( act.ApModel )-1 ].ItsJITCompiler = append(act.ApModel[ len( act.ApModel )-1 ].ItsJITCompiler, st)	// net5.unit:94, g_structh.act:403
	name,_ := st.Names["jit_compiler"]
	s := strconv.Itoa(st.Kparentp) + "_JITCompiler_" + name	// net5.unit:1074, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApJITCompiler = append(act.ApJITCompiler, st)
	return 0
}

func (me KpJITCompiler) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // net5.unit:94, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // net5.unit:1069, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApJITCompiler[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,JITCompiler > net5.unit:1069, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,JITCompiler > net5.unit:1069, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpJITCompiler) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // net5.unit:94, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApModel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for JITCompiler %s,%s > net5.unit:1069, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTargetRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpTargetRule) TypeName() string {
    return me.Comp
}
func (me KpTargetRule) GetLineNo() string {
	return me.LineNo
}

func loadTargetRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTargetRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTargetRule)
	st.LineNo = lno
	st.Comp = "TargetRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["target"]
	act.index["TargetRule_" + name] = st.Me;
	st.MyName = name
	act.ApTargetRule = append(act.ApTargetRule, st)
	return 0
}

func (me KpTargetRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Config_target" && len(va) > 1) { // net5.unit:639, g_structh.act:698
		for _, st := range glob.Dats.ApConfig {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // net5.unit:1084, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTargetRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TargetRule > net5.unit:1084, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TargetRule > net5.unit:1084, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTargetRule) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Config_target") { // net5.unit:639, g_structh.act:583
		for _, st := range glob.Dats.ApConfig {
			if (st.Ktargetp == me.Me) {
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
	        fmt.Printf("?No its %s for TargetRule %s,%s > net5.unit:1084, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDtypeRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDtypeRule) TypeName() string {
    return me.Comp
}
func (me KpDtypeRule) GetLineNo() string {
	return me.LineNo
}

func loadDtypeRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDtypeRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDtypeRule)
	st.LineNo = lno
	st.Comp = "DtypeRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["dtype"]
	act.index["DtypeRule_" + name] = st.Me;
	st.MyName = name
	act.ApDtypeRule = append(act.ApDtypeRule, st)
	return 0
}

func (me KpDtypeRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Tensor_dtype" && len(va) > 1) { // net5.unit:160, g_structh.act:698
		for _, st := range glob.Dats.ApTensor {
			if (st.Kdtypep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ArgRoleRule_dtype" && len(va) > 1) { // net5.unit:629, g_structh.act:698
		for _, st := range glob.Dats.ApArgRoleRule {
			if (st.Kdtypep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "GraphTensor_dtype" && len(va) > 1) { // net5.unit:711, g_structh.act:698
		for _, st := range glob.Dats.ApGraphTensor {
			if (st.Kdtypep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // net5.unit:1093, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDtypeRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DtypeRule > net5.unit:1093, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DtypeRule > net5.unit:1093, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDtypeRule) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Tensor_dtype") { // net5.unit:160, g_structh.act:583
		for _, st := range glob.Dats.ApTensor {
			if (st.Kdtypep == me.Me) {
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
	if (va[0] == "ArgRoleRule_dtype") { // net5.unit:629, g_structh.act:583
		for _, st := range glob.Dats.ApArgRoleRule {
			if (st.Kdtypep == me.Me) {
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
	if (va[0] == "GraphTensor_dtype") { // net5.unit:711, g_structh.act:583
		for _, st := range glob.Dats.ApGraphTensor {
			if (st.Kdtypep == me.Me) {
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
	        fmt.Printf("?No its %s for DtypeRule %s,%s > net5.unit:1093, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFlagRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpFlagRule) TypeName() string {
    return me.Comp
}
func (me KpFlagRule) GetLineNo() string {
	return me.LineNo
}

func loadFlagRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFlagRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFlagRule)
	st.LineNo = lno
	st.Comp = "FlagRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["flag"]
	act.index["FlagRule_" + name] = st.Me;
	st.MyName = name
	act.ApFlagRule = append(act.ApFlagRule, st)
	return 0
}

func (me KpFlagRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Optimization_flags" && len(va) > 1) { // net5.unit:57, g_structh.act:698
		for _, st := range glob.Dats.ApOptimization {
			if (st.Kflagsp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Config_opt_flags" && len(va) > 1) { // net5.unit:641, g_structh.act:698
		for _, st := range glob.Dats.ApConfig {
			if (st.Kopt_flagsp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // net5.unit:1102, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFlagRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FlagRule > net5.unit:1102, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FlagRule > net5.unit:1102, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFlagRule) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Optimization_flags") { // net5.unit:57, g_structh.act:583
		for _, st := range glob.Dats.ApOptimization {
			if (st.Kflagsp == me.Me) {
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
	if (va[0] == "Config_opt_flags") { // net5.unit:641, g_structh.act:583
		for _, st := range glob.Dats.ApConfig {
			if (st.Kopt_flagsp == me.Me) {
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
	        fmt.Printf("?No its %s for FlagRule %s,%s > net5.unit:1102, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDistributionRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDistributionRule) TypeName() string {
    return me.Comp
}
func (me KpDistributionRule) GetLineNo() string {
	return me.LineNo
}

func loadDistributionRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDistributionRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDistributionRule)
	st.LineNo = lno
	st.Comp = "DistributionRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["distribution"]
	act.index["DistributionRule_" + name] = st.Me;
	st.MyName = name
	act.ApDistributionRule = append(act.ApDistributionRule, st)
	return 0
}

func (me KpDistributionRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:1110, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDistributionRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DistributionRule > net5.unit:1110, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DistributionRule > net5.unit:1110, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDistributionRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DistributionRule %s,%s > net5.unit:1110, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchStrategyRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpSearchStrategyRule) TypeName() string {
    return me.Comp
}
func (me KpSearchStrategyRule) GetLineNo() string {
	return me.LineNo
}

func loadSearchStrategyRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchStrategyRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchStrategyRule)
	st.LineNo = lno
	st.Comp = "SearchStrategyRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["strategy"]
	act.index["SearchStrategyRule_" + name] = st.Me;
	st.MyName = name
	act.ApSearchStrategyRule = append(act.ApSearchStrategyRule, st)
	return 0
}

func (me KpSearchStrategyRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:1120, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchStrategyRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchStrategyRule > net5.unit:1120, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchStrategyRule > net5.unit:1120, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchStrategyRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for SearchStrategyRule %s,%s > net5.unit:1120, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDynamicGraphRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDynamicGraphRule) TypeName() string {
    return me.Comp
}
func (me KpDynamicGraphRule) GetLineNo() string {
	return me.LineNo
}

func loadDynamicGraphRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDynamicGraphRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDynamicGraphRule)
	st.LineNo = lno
	st.Comp = "DynamicGraphRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["graph_rule"]
	act.index["DynamicGraphRule_" + name] = st.Me;
	st.MyName = name
	act.ApDynamicGraphRule = append(act.ApDynamicGraphRule, st)
	return 0
}

func (me KpDynamicGraphRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:1130, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicGraphRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DynamicGraphRule > net5.unit:1130, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DynamicGraphRule > net5.unit:1130, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDynamicGraphRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DynamicGraphRule %s,%s > net5.unit:1130, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCompatibilityNote struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpCompatibilityNote) TypeName() string {
    return me.Comp
}
func (me KpCompatibilityNote) GetLineNo() string {
	return me.LineNo
}

func loadCompatibilityNote(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCompatibilityNote)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCompatibilityNote)
	st.LineNo = lno
	st.Comp = "CompatibilityNote";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["note"]
	act.index["CompatibilityNote_" + name] = st.Me;
	st.MyName = name
	act.ApCompatibilityNote = append(act.ApCompatibilityNote, st)
	return 0
}

func (me KpCompatibilityNote) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // net5.unit:1144, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCompatibilityNote[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CompatibilityNote > net5.unit:1144, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CompatibilityNote > net5.unit:1144, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCompatibilityNote) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for CompatibilityNote %s,%s > net5.unit:1144, g_structh.act:209?", va[0], lno, me.LineNo)
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

