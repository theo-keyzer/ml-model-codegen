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
	if (va[0] == "Project_domain" && len(va) > 1) { // tsu.unit:422, g_structh.act:698
		for _, st := range glob.Dats.ApProject {
			if (st.Kdomainp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:10, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Domain > tsu.unit:10, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Domain > tsu.unit:10, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDomain) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Project_domain") { // tsu.unit:422, g_structh.act:583
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
	        fmt.Printf("?No its %s for Domain %s,%s > tsu.unit:10, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	ItsTSU [] *KpTSU 
	Childs [] Kp
}

func (me KpHardware) TypeName() string {
    return me.Comp
}
func (me KpHardware) GetLineNo() string {
	return me.LineNo
}

func loadHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["hardware"]
	act.index["Hardware_" + name] = st.Me;
	st.MyName = name
	act.ApHardware = append(act.ApHardware, st)
	return 0
}

func (me KpHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Kernel_hardware" && len(va) > 1) { // tsu.unit:51, g_structh.act:698
		for _, st := range glob.Dats.ApKernel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "TSUKernel_hardware" && len(va) > 1) { // tsu.unit:65, g_structh.act:698
		for _, st := range glob.Dats.ApTSUKernel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Optimization_target" && len(va) > 1) { // tsu.unit:101, g_structh.act:698
		for _, st := range glob.Dats.ApOptimization {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Fusion_hardware" && len(va) > 1) { // tsu.unit:113, g_structh.act:698
		for _, st := range glob.Dats.ApFusion {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Model_hardware" && len(va) > 1) { // tsu.unit:167, g_structh.act:698
		for _, st := range glob.Dats.ApModel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "TSUCompilation_hardware" && len(va) > 1) { // tsu.unit:291, g_structh.act:698
		for _, st := range glob.Dats.ApTSUCompilation {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Config_target" && len(va) > 1) { // tsu.unit:322, g_structh.act:698
		for _, st := range glob.Dats.ApConfig {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "TargetConfig_hardware" && len(va) > 1) { // tsu.unit:431, g_structh.act:698
		for _, st := range glob.Dats.ApTargetConfig {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:18, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Hardware > tsu.unit:18, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Hardware > tsu.unit:18, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TSU" { // tsu.unit:30, g_structh.act:676
		for _, st := range me.ItsTSU {
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
	if (va[0] == "Kernel_hardware") { // tsu.unit:51, g_structh.act:583
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
	if (va[0] == "TSUKernel_hardware") { // tsu.unit:65, g_structh.act:583
		for _, st := range glob.Dats.ApTSUKernel {
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
	if (va[0] == "Optimization_target") { // tsu.unit:101, g_structh.act:583
		for _, st := range glob.Dats.ApOptimization {
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
	if (va[0] == "Fusion_hardware") { // tsu.unit:113, g_structh.act:583
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
	if (va[0] == "Model_hardware") { // tsu.unit:167, g_structh.act:583
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
	if (va[0] == "TSUCompilation_hardware") { // tsu.unit:291, g_structh.act:583
		for _, st := range glob.Dats.ApTSUCompilation {
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
	if (va[0] == "Config_target") { // tsu.unit:322, g_structh.act:583
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
	if (va[0] == "TargetConfig_hardware") { // tsu.unit:431, g_structh.act:583
		for _, st := range glob.Dats.ApTargetConfig {
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
	        fmt.Printf("?No its %s for Hardware %s,%s > tsu.unit:18, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTSU struct {
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

func (me KpTSU) TypeName() string {
    return me.Comp
}
func (me KpTSU) GetLineNo() string {
	return me.LineNo
}

func loadTSU(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTSU)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTSU)
	st.LineNo = lno
	st.Comp = "TSU";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TSU has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TSU under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsTSU = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsTSU, st)	// tsu.unit:18, g_structh.act:403
	name,_ := st.Names["tsu_id"]
	s := strconv.Itoa(st.Kparentp) + "_TSU_" + name	// tsu.unit:34, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTSU = append(act.ApTSU, st)
	return 0
}

func (me KpTSU) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:18, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:30, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTSU[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TSU > tsu.unit:30, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TSU > tsu.unit:30, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTSU) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:18, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "TSUKernel_tsu_target") { // tsu.unit:66, g_structh.act:583
		for _, st := range glob.Dats.ApTSUKernel {
			if (st.Ktsu_targetp == me.Me) {
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
	if (va[0] == "TSUCompilation_target") { // tsu.unit:292, g_structh.act:583
		for _, st := range glob.Dats.ApTSUCompilation {
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
	        fmt.Printf("?No its %s for TSU %s,%s > tsu.unit:30, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Khardwarep int
	ItsTSUKernel [] *KpTSUKernel 
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
	st.Khardwarep = -1
	name,_ := st.Names["kernel"]
	act.index["Kernel_" + name] = st.Me;
	st.MyName = name
	act.ApKernel = append(act.ApKernel, st)
	return 0
}

func (me KpKernel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // tsu.unit:51, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "TSUKernel_copy_kernel" && len(va) > 1) { // tsu.unit:64, g_structh.act:698
		for _, st := range glob.Dats.ApTSUKernel {
			if (st.Kcopy_kernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Fusion_fused_kernel" && len(va) > 1) { // tsu.unit:112, g_structh.act:698
		for _, st := range glob.Dats.ApFusion {
			if (st.Kfused_kernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_kernel" && len(va) > 1) { // tsu.unit:206, g_structh.act:698
		for _, st := range glob.Dats.ApOp {
			if (st.Kkernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:46, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Kernel > tsu.unit:46, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Kernel > tsu.unit:46, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TSUKernel" { // tsu.unit:59, g_structh.act:676
		for _, st := range me.ItsTSUKernel {
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
	if va[0] == "KernelParam" { // tsu.unit:73, g_structh.act:676
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
	if va[0] == "KernelOp" { // tsu.unit:83, g_structh.act:676
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
	if (va[0] == "TSUKernel_copy_kernel") { // tsu.unit:64, g_structh.act:583
		for _, st := range glob.Dats.ApTSUKernel {
			if (st.Kcopy_kernelp == me.Me) {
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
	if (va[0] == "Fusion_fused_kernel") { // tsu.unit:112, g_structh.act:583
		for _, st := range glob.Dats.ApFusion {
			if (st.Kfused_kernelp == me.Me) {
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
	if (va[0] == "Op_kernel") { // tsu.unit:206, g_structh.act:583
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
	        fmt.Printf("?No its %s for Kernel %s,%s > tsu.unit:46, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTSUKernel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kcopy_kernelp int
	Khardwarep int
	Ktsu_targetp int
}

func (me KpTSUKernel) TypeName() string {
    return me.Comp
}
func (me KpTSUKernel) GetLineNo() string {
	return me.LineNo
}

func loadTSUKernel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTSUKernel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTSUKernel)
	st.LineNo = lno
	st.Comp = "TSUKernel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcopy_kernelp = -1
	st.Khardwarep = -1
	st.Ktsu_targetp = -1
	st.Kparentp = len( act.ApKernel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TSUKernel has no Kernel parent\n") ;
		return 1
	}
	st.Parent = act.ApKernel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TSUKernel under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApKernel[ len( act.ApKernel )-1 ].Childs = append(act.ApKernel[ len( act.ApKernel )-1 ].Childs, st)
	act.ApKernel[ len( act.ApKernel )-1 ].ItsTSUKernel = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsTSUKernel, st)	// tsu.unit:46, g_structh.act:403
	name,_ := st.Names["tsu_kernel"]
	s := strconv.Itoa(st.Kparentp) + "_TSUKernel_" + name	// tsu.unit:63, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTSUKernel = append(act.ApTSUKernel, st)
	return 0
}

func (me KpTSUKernel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "copy_kernel" { // tsu.unit:64, g_structh.act:609
		if (me.Kcopy_kernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kcopy_kernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // tsu.unit:65, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "tsu_target" { // tsu.unit:66, g_structh.act:609
		if (me.Ktsu_targetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTSU[ me.Ktsu_targetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:46, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:59, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTSUKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TSUKernel > tsu.unit:59, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TSUKernel > tsu.unit:59, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTSUKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:46, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "copy_kernel" {
		if me.Kcopy_kernelp >= 0 {
			st := glob.Dats.ApKernel[ me.Kcopy_kernelp ]
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
	if va[0] == "tsu_target" {
		if me.Ktsu_targetp >= 0 {
			st := glob.Dats.ApTSU[ me.Ktsu_targetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TSUKernel %s,%s > tsu.unit:59, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelParam = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelParam, st)	// tsu.unit:46, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_KernelParam_" + name	// tsu.unit:78, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApKernelParam = append(act.ApKernelParam, st)
	return 0
}

func (me KpKernelParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:46, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:73, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelParam > tsu.unit:73, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelParam > tsu.unit:73, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:46, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for KernelParam %s,%s > tsu.unit:73, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelOp = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelOp, st)	// tsu.unit:46, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_KernelOp_" + name	// tsu.unit:89, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApKernelOp = append(act.ApKernelOp, st)
	return 0
}

func (me KpKernelOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:46, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:83, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelOp > tsu.unit:83, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelOp > tsu.unit:83, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:46, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Op_kernel_op") { // tsu.unit:207, g_structh.act:583
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
	        fmt.Printf("?No its %s for KernelOp %s,%s > tsu.unit:83, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Ktargetp int
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
	st.Ktargetp = -1
	act.ApOptimization = append(act.ApOptimization, st)
	return 0
}

func (me KpOptimization) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target" { // tsu.unit:101, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu.unit:97, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimization[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Optimization > tsu.unit:97, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Optimization > tsu.unit:97, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOptimization) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "target" {
		if me.Ktargetp >= 0 {
			st := glob.Dats.ApHardware[ me.Ktargetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Optimization %s,%s > tsu.unit:97, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kfused_kernelp int
	Khardwarep int
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
	st.Kfused_kernelp = -1
	st.Khardwarep = -1
	name,_ := st.Names["fusion"]
	act.index["Fusion_" + name] = st.Me;
	st.MyName = name
	act.ApFusion = append(act.ApFusion, st)
	return 0
}

func (me KpFusion) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "fused_kernel" { // tsu.unit:112, g_structh.act:609
		if (me.Kfused_kernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kfused_kernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // tsu.unit:113, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu.unit:106, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusion[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Fusion > tsu.unit:106, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Fusion > tsu.unit:106, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFusion) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "fused_kernel" {
		if me.Kfused_kernelp >= 0 {
			st := glob.Dats.ApKernel[ me.Kfused_kernelp ]
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
	        fmt.Printf("?No its %s for Fusion %s,%s > tsu.unit:106, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFramework struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	ItsTHRML [] *KpTHRML 
	Childs [] Kp
}

func (me KpFramework) TypeName() string {
    return me.Comp
}
func (me KpFramework) GetLineNo() string {
	return me.LineNo
}

func loadFramework(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFramework)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFramework)
	st.LineNo = lno
	st.Comp = "Framework";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["framework"]
	act.index["Framework_" + name] = st.Me;
	st.MyName = name
	act.ApFramework = append(act.ApFramework, st)
	return 0
}

func (me KpFramework) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Model_framework" && len(va) > 1) { // tsu.unit:168, g_structh.act:698
		for _, st := range glob.Dats.ApModel {
			if (st.Kframeworkp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:120, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFramework[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Framework > tsu.unit:120, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Framework > tsu.unit:120, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFramework) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "THRML" { // tsu.unit:130, g_structh.act:676
		for _, st := range me.ItsTHRML {
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
	if (va[0] == "Model_framework") { // tsu.unit:168, g_structh.act:583
		for _, st := range glob.Dats.ApModel {
			if (st.Kframeworkp == me.Me) {
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
	        fmt.Printf("?No its %s for Framework %s,%s > tsu.unit:120, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTHRML struct {
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

func (me KpTHRML) TypeName() string {
    return me.Comp
}
func (me KpTHRML) GetLineNo() string {
	return me.LineNo
}

func loadTHRML(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTHRML)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTHRML)
	st.LineNo = lno
	st.Comp = "THRML";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApFramework ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " THRML has no Framework parent\n") ;
		return 1
	}
	st.Parent = act.ApFramework[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " THRML under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApFramework[ len( act.ApFramework )-1 ].Childs = append(act.ApFramework[ len( act.ApFramework )-1 ].Childs, st)
	act.ApFramework[ len( act.ApFramework )-1 ].ItsTHRML = append(act.ApFramework[ len( act.ApFramework )-1 ].ItsTHRML, st)	// tsu.unit:120, g_structh.act:403
	name,_ := st.Names["thrml_id"]
	s := strconv.Itoa(st.Kparentp) + "_THRML_" + name	// tsu.unit:134, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTHRML = append(act.ApTHRML, st)
	return 0
}

func (me KpTHRML) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:120, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFramework[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:130, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTHRML[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,THRML > tsu.unit:130, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,THRML > tsu.unit:130, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTHRML) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:120, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApFramework[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for THRML %s,%s > tsu.unit:130, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPGMSchema struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpPGMSchema) TypeName() string {
    return me.Comp
}
func (me KpPGMSchema) GetLineNo() string {
	return me.LineNo
}

func loadPGMSchema(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPGMSchema)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPGMSchema)
	st.LineNo = lno
	st.Comp = "PGMSchema";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["schema"]
	act.index["PGMSchema_" + name] = st.Me;
	st.MyName = name
	act.ApPGMSchema = append(act.ApPGMSchema, st)
	return 0
}

func (me KpPGMSchema) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Model_pgm_schema" && len(va) > 1) { // tsu.unit:169, g_structh.act:698
		for _, st := range glob.Dats.ApModel {
			if (st.Kpgm_schemap == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:143, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPGMSchema[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PGMSchema > tsu.unit:143, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PGMSchema > tsu.unit:143, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPGMSchema) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Model_pgm_schema") { // tsu.unit:169, g_structh.act:583
		for _, st := range glob.Dats.ApModel {
			if (st.Kpgm_schemap == me.Me) {
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
	        fmt.Printf("?No its %s for PGMSchema %s,%s > tsu.unit:143, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Khardwarep int
	Kframeworkp int
	Kpgm_schemap int
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
	st.Khardwarep = -1
	st.Kframeworkp = -1
	st.Kpgm_schemap = -1
	name,_ := st.Names["model"]
	act.index["Model_" + name] = st.Me;
	st.MyName = name
	act.ApModel = append(act.ApModel, st)
	return 0
}

func (me KpModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // tsu.unit:167, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "framework" { // tsu.unit:168, g_structh.act:609
		if (me.Kframeworkp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFramework[ me.Kframeworkp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "pgm_schema" { // tsu.unit:169, g_structh.act:609
		if (me.Kpgm_schemap >= 0 && len(va) > 1) {
			return( glob.Dats.ApPGMSchema[ me.Kpgm_schemap ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "TSUCompilation_source" && len(va) > 1) { // tsu.unit:290, g_structh.act:698
		for _, st := range glob.Dats.ApTSUCompilation {
			if (st.Ksourcep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Validation_target" && len(va) > 1) { // tsu.unit:356, g_structh.act:698
		for _, st := range glob.Dats.ApValidation {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Project_model" && len(va) > 1) { // tsu.unit:423, g_structh.act:698
		for _, st := range glob.Dats.ApProject {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:161, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Model > tsu.unit:161, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Model > tsu.unit:161, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpModel) DoIts(glob *GlobT, va []string, lno string) int {
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
	if va[0] == "framework" {
		if me.Kframeworkp >= 0 {
			st := glob.Dats.ApFramework[ me.Kframeworkp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "pgm_schema" {
		if me.Kpgm_schemap >= 0 {
			st := glob.Dats.ApPGMSchema[ me.Kpgm_schemap ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "TSUCompilation_source") { // tsu.unit:290, g_structh.act:583
		for _, st := range glob.Dats.ApTSUCompilation {
			if (st.Ksourcep == me.Me) {
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
	if (va[0] == "Validation_target") { // tsu.unit:356, g_structh.act:583
		for _, st := range glob.Dats.ApValidation {
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
	if (va[0] == "Project_model") { // tsu.unit:423, g_structh.act:583
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
	        fmt.Printf("?No its %s for Model %s,%s > tsu.unit:161, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kdistributionp int
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
	st.Kdistributionp = -1
	name,_ := st.Names["tensor"]
	act.index["Tensor_" + name] = st.Me;
	st.MyName = name
	act.ApTensor = append(act.ApTensor, st)
	return 0
}

func (me KpTensor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "distribution" { // tsu.unit:183, g_structh.act:609
		if (me.Kdistributionp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDistributionRule[ me.Kdistributionp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Arg_tensor" && len(va) > 1) { // tsu.unit:215, g_structh.act:698
		for _, st := range glob.Dats.ApArg {
			if (st.Ktensorp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "EnergyFunction_params" && len(va) > 1) { // tsu.unit:263, g_structh.act:698
		for _, st := range glob.Dats.ApEnergyFunction {
			if (st.Kparamsp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "EnergyFactor_variables" && len(va) > 1) { // tsu.unit:273, g_structh.act:698
		for _, st := range glob.Dats.ApEnergyFactor {
			if (st.Kvariablesp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "EnergyFactor_param_tensor" && len(va) > 1) { // tsu.unit:276, g_structh.act:698
		for _, st := range glob.Dats.ApEnergyFactor {
			if (st.Kparam_tensorp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:173, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Tensor > tsu.unit:173, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Tensor > tsu.unit:173, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTensor) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "distribution" {
		if me.Kdistributionp >= 0 {
			st := glob.Dats.ApDistributionRule[ me.Kdistributionp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Arg_tensor") { // tsu.unit:215, g_structh.act:583
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
	if (va[0] == "EnergyFunction_params") { // tsu.unit:263, g_structh.act:583
		for _, st := range glob.Dats.ApEnergyFunction {
			if (st.Kparamsp == me.Me) {
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
	if (va[0] == "EnergyFactor_variables") { // tsu.unit:273, g_structh.act:583
		for _, st := range glob.Dats.ApEnergyFactor {
			if (st.Kvariablesp == me.Me) {
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
	if (va[0] == "EnergyFactor_param_tensor") { // tsu.unit:276, g_structh.act:583
		for _, st := range glob.Dats.ApEnergyFactor {
			if (st.Kparam_tensorp == me.Me) {
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
	        fmt.Printf("?No its %s for Tensor %s,%s > tsu.unit:173, g_structh.act:209?", va[0], lno, me.LineNo)
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
	name,_ := st.Names["layer"]
	act.index["Layer_" + name] = st.Me;
	st.MyName = name
	act.ApLayer = append(act.ApLayer, st)
	return 0
}

func (me KpLayer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Schedule_layer" && len(va) > 1) { // tsu.unit:332, g_structh.act:698
		for _, st := range glob.Dats.ApSchedule {
			if (st.Klayerp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:186, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Layer > tsu.unit:186, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Layer > tsu.unit:186, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpLayer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Op" { // tsu.unit:198, g_structh.act:676
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
	if (va[0] == "Schedule_layer") { // tsu.unit:332, g_structh.act:583
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
	        fmt.Printf("?No its %s for Layer %s,%s > tsu.unit:186, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsSamplingOp [] *KpSamplingOp 
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
	act.ApLayer[ len( act.ApLayer )-1 ].ItsOp = append(act.ApLayer[ len( act.ApLayer )-1 ].ItsOp, st)	// tsu.unit:186, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_Op_" + name	// tsu.unit:202, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOp = append(act.ApOp, st)
	return 0
}

func (me KpOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "op_type" { // tsu.unit:203, g_structh.act:619
		if (me.Kop_typep >= 0 && len(va) > 1) {
			return( me.Childs[ me.Kop_typep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel" { // tsu.unit:206, g_structh.act:609
		if (me.Kkernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kkernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel_op" { // tsu.unit:207, g_structh.act:609
		if (me.Kkernel_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernelOp[ me.Kkernel_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:186, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:198, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Op > tsu.unit:198, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Op > tsu.unit:198, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Arg" { // tsu.unit:209, g_structh.act:676
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
	if va[0] == "SamplingOp" { // tsu.unit:220, g_structh.act:676
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
	if va[0] == "parent" { // tsu.unit:186, g_structh.act:557
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
	if (va[0] == "Schedule_op") { // tsu.unit:333, g_structh.act:583
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
	        fmt.Printf("?No its %s for Op %s,%s > tsu.unit:198, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApOp[ len( act.ApOp )-1 ].ItsArg = append(act.ApOp[ len( act.ApOp )-1 ].ItsArg, st)	// tsu.unit:198, g_structh.act:403
	name,_ := st.Names["arg"]
	s := strconv.Itoa(st.Kparentp) + "_Arg_" + name	// tsu.unit:213, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArg = append(act.ApArg, st)
	return 0
}

func (me KpArg) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "tensor" { // tsu.unit:215, g_structh.act:609
		if (me.Ktensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:198, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:209, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArg[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Arg > tsu.unit:209, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Arg > tsu.unit:209, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArg) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:198, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
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
	        fmt.Printf("?No its %s for Arg %s,%s > tsu.unit:209, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kenergy_fn_refp int
	ItsTSUSamplingOp [] *KpTSUSamplingOp 
	ItsBlockGibbsOp [] *KpBlockGibbsOp 
	Childs [] Kp
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
	st.Kenergy_fn_refp = -1
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
	act.ApOp[ len( act.ApOp )-1 ].ItsSamplingOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsSamplingOp, st)	// tsu.unit:198, g_structh.act:403
	name,_ := st.Names["sampling_op"]
	s := strconv.Itoa(st.Kparentp) + "_SamplingOp_" + name	// tsu.unit:224, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSamplingOp = append(act.ApSamplingOp, st)
	return 0
}

func (me KpSamplingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "energy_fn_ref" { // tsu.unit:231, g_structh.act:609
		if (me.Kenergy_fn_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyFunction[ me.Kenergy_fn_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:198, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:220, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSamplingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SamplingOp > tsu.unit:220, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SamplingOp > tsu.unit:220, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSamplingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TSUSamplingOp" { // tsu.unit:234, g_structh.act:676
		for _, st := range me.ItsTSUSamplingOp {
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
	if va[0] == "BlockGibbsOp" { // tsu.unit:244, g_structh.act:676
		for _, st := range me.ItsBlockGibbsOp {
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
	if va[0] == "parent" { // tsu.unit:198, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "energy_fn_ref" {
		if me.Kenergy_fn_refp >= 0 {
			st := glob.Dats.ApEnergyFunction[ me.Kenergy_fn_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SamplingOp %s,%s > tsu.unit:220, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTSUSamplingOp struct {
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

func (me KpTSUSamplingOp) TypeName() string {
    return me.Comp
}
func (me KpTSUSamplingOp) GetLineNo() string {
	return me.LineNo
}

func loadTSUSamplingOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTSUSamplingOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTSUSamplingOp)
	st.LineNo = lno
	st.Comp = "TSUSamplingOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSamplingOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TSUSamplingOp has no SamplingOp parent\n") ;
		return 1
	}
	st.Parent = act.ApSamplingOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TSUSamplingOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].Childs = append(act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].Childs, st)
	act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].ItsTSUSamplingOp = append(act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].ItsTSUSamplingOp, st)	// tsu.unit:220, g_structh.act:403
	name,_ := st.Names["tsu_op"]
	s := strconv.Itoa(st.Kparentp) + "_TSUSamplingOp_" + name	// tsu.unit:238, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTSUSamplingOp = append(act.ApTSUSamplingOp, st)
	return 0
}

func (me KpTSUSamplingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:220, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSamplingOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:234, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTSUSamplingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TSUSamplingOp > tsu.unit:234, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TSUSamplingOp > tsu.unit:234, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTSUSamplingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:220, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSamplingOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TSUSamplingOp %s,%s > tsu.unit:234, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBlockGibbsOp struct {
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

func (me KpBlockGibbsOp) TypeName() string {
    return me.Comp
}
func (me KpBlockGibbsOp) GetLineNo() string {
	return me.LineNo
}

func loadBlockGibbsOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpBlockGibbsOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBlockGibbsOp)
	st.LineNo = lno
	st.Comp = "BlockGibbsOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSamplingOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " BlockGibbsOp has no SamplingOp parent\n") ;
		return 1
	}
	st.Parent = act.ApSamplingOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " BlockGibbsOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].Childs = append(act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].Childs, st)
	act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].ItsBlockGibbsOp = append(act.ApSamplingOp[ len( act.ApSamplingOp )-1 ].ItsBlockGibbsOp, st)	// tsu.unit:220, g_structh.act:403
	name,_ := st.Names["block_op"]
	s := strconv.Itoa(st.Kparentp) + "_BlockGibbsOp_" + name	// tsu.unit:248, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBlockGibbsOp = append(act.ApBlockGibbsOp, st)
	return 0
}

func (me KpBlockGibbsOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:220, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSamplingOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:244, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBlockGibbsOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,BlockGibbsOp > tsu.unit:244, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,BlockGibbsOp > tsu.unit:244, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBlockGibbsOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:220, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSamplingOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for BlockGibbsOp %s,%s > tsu.unit:244, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparamsp int
}

func (me KpEnergyFunction) TypeName() string {
    return me.Comp
}
func (me KpEnergyFunction) GetLineNo() string {
	return me.LineNo
}

func loadEnergyFunction(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kparamsp = -1
	name,_ := st.Names["energy_fn"]
	act.index["EnergyFunction_" + name] = st.Me;
	st.MyName = name
	act.ApEnergyFunction = append(act.ApEnergyFunction, st)
	return 0
}

func (me KpEnergyFunction) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "params" { // tsu.unit:263, g_structh.act:609
		if (me.Kparamsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kparamsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "SamplingOp_energy_fn_ref" && len(va) > 1) { // tsu.unit:231, g_structh.act:698
		for _, st := range glob.Dats.ApSamplingOp {
			if (st.Kenergy_fn_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:256, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyFunction[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnergyFunction > tsu.unit:256, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnergyFunction > tsu.unit:256, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnergyFunction) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "params" {
		if me.Kparamsp >= 0 {
			st := glob.Dats.ApTensor[ me.Kparamsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "SamplingOp_energy_fn_ref") { // tsu.unit:231, g_structh.act:583
		for _, st := range glob.Dats.ApSamplingOp {
			if (st.Kenergy_fn_refp == me.Me) {
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
	        fmt.Printf("?No its %s for EnergyFunction %s,%s > tsu.unit:256, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnergyFactor struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kvariablesp int
	Kparam_tensorp int
}

func (me KpEnergyFactor) TypeName() string {
    return me.Comp
}
func (me KpEnergyFactor) GetLineNo() string {
	return me.LineNo
}

func loadEnergyFactor(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEnergyFactor)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnergyFactor)
	st.LineNo = lno
	st.Comp = "EnergyFactor";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kvariablesp = -1
	st.Kparam_tensorp = -1
	name,_ := st.Names["factor"]
	act.index["EnergyFactor_" + name] = st.Me;
	st.MyName = name
	act.ApEnergyFactor = append(act.ApEnergyFactor, st)
	return 0
}

func (me KpEnergyFactor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "variables" { // tsu.unit:273, g_structh.act:609
		if (me.Kvariablesp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kvariablesp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "param_tensor" { // tsu.unit:276, g_structh.act:609
		if (me.Kparam_tensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Kparam_tensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu.unit:267, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyFactor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnergyFactor > tsu.unit:267, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnergyFactor > tsu.unit:267, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnergyFactor) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "variables" {
		if me.Kvariablesp >= 0 {
			st := glob.Dats.ApTensor[ me.Kvariablesp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "param_tensor" {
		if me.Kparam_tensorp >= 0 {
			st := glob.Dats.ApTensor[ me.Kparam_tensorp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for EnergyFactor %s,%s > tsu.unit:267, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTSUCompilation struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Ksourcep int
	Khardwarep int
	Ktargetp int
}

func (me KpTSUCompilation) TypeName() string {
    return me.Comp
}
func (me KpTSUCompilation) GetLineNo() string {
	return me.LineNo
}

func loadTSUCompilation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTSUCompilation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTSUCompilation)
	st.LineNo = lno
	st.Comp = "TSUCompilation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ksourcep = -1
	st.Khardwarep = -1
	st.Ktargetp = -1
	name,_ := st.Names["plan_id"]
	act.index["TSUCompilation_" + name] = st.Me;
	st.MyName = name
	act.ApTSUCompilation = append(act.ApTSUCompilation, st)
	return 0
}

func (me KpTSUCompilation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "source" { // tsu.unit:290, g_structh.act:609
		if (me.Ksourcep >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Ksourcep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // tsu.unit:291, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "target" { // tsu.unit:292, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTSU[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu.unit:285, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTSUCompilation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TSUCompilation > tsu.unit:285, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TSUCompilation > tsu.unit:285, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTSUCompilation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "source" {
		if me.Ksourcep >= 0 {
			st := glob.Dats.ApModel[ me.Ksourcep ]
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
	if va[0] == "target" {
		if me.Ktargetp >= 0 {
			st := glob.Dats.ApTSU[ me.Ktargetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TSUCompilation %s,%s > tsu.unit:285, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTSUSubstrateModel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpTSUSubstrateModel) TypeName() string {
    return me.Comp
}
func (me KpTSUSubstrateModel) GetLineNo() string {
	return me.LineNo
}

func loadTSUSubstrateModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTSUSubstrateModel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTSUSubstrateModel)
	st.LineNo = lno
	st.Comp = "TSUSubstrateModel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["substrate"]
	act.index["TSUSubstrateModel_" + name] = st.Me;
	st.MyName = name
	act.ApTSUSubstrateModel = append(act.ApTSUSubstrateModel, st)
	return 0
}

func (me KpTSUSubstrateModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu.unit:300, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTSUSubstrateModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TSUSubstrateModel > tsu.unit:300, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TSUSubstrateModel > tsu.unit:300, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTSUSubstrateModel) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for TSUSubstrateModel %s,%s > tsu.unit:300, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Ktargetp int
	ItsSchedule [] *KpSchedule 
	ItsAutoTuneConfig [] *KpAutoTuneConfig 
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
	name,_ := st.Names["config"]
	act.index["Config_" + name] = st.Me;
	st.MyName = name
	act.ApConfig = append(act.ApConfig, st)
	return 0
}

func (me KpConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target" { // tsu.unit:322, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "TargetConfig_config" && len(va) > 1) { // tsu.unit:433, g_structh.act:698
		for _, st := range glob.Dats.ApTargetConfig {
			if (st.Kconfigp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchSpace_target" && len(va) > 1) { // tsu-auto.unit:10, g_structh.act:698
		for _, st := range glob.Dats.ApSearchSpace {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:317, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Config > tsu.unit:317, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Config > tsu.unit:317, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Schedule" { // tsu.unit:327, g_structh.act:676
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
	if va[0] == "AutoTuneConfig" { // tsu-auto.unit:159, g_structh.act:676
		for _, st := range me.ItsAutoTuneConfig {
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
	if va[0] == "target" {
		if me.Ktargetp >= 0 {
			st := glob.Dats.ApHardware[ me.Ktargetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "TargetConfig_config") { // tsu.unit:433, g_structh.act:583
		for _, st := range glob.Dats.ApTargetConfig {
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
	if (va[0] == "SearchSpace_target") { // tsu-auto.unit:10, g_structh.act:583
		for _, st := range glob.Dats.ApSearchSpace {
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
	        fmt.Printf("?No its %s for Config %s,%s > tsu.unit:317, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Klayerp int
	Kopp int
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
	st.Klayerp = -1
	st.Kopp = -1
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
	act.ApConfig[ len( act.ApConfig )-1 ].ItsSchedule = append(act.ApConfig[ len( act.ApConfig )-1 ].ItsSchedule, st)	// tsu.unit:317, g_structh.act:403
	act.ApSchedule = append(act.ApSchedule, st)
	return 0
}

func (me KpSchedule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "layer" { // tsu.unit:332, g_structh.act:609
		if (me.Klayerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "op" { // tsu.unit:333, g_structh.act:609
		if (me.Kopp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kopp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:317, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:327, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSchedule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Schedule > tsu.unit:327, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Schedule > tsu.unit:327, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSchedule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:317, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApConfig[ me.Kparentp ]
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
	        fmt.Printf("?No its %s for Schedule %s,%s > tsu.unit:327, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpThermodynamicSimulation struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpThermodynamicSimulation) TypeName() string {
    return me.Comp
}
func (me KpThermodynamicSimulation) GetLineNo() string {
	return me.LineNo
}

func loadThermodynamicSimulation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpThermodynamicSimulation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApThermodynamicSimulation)
	st.LineNo = lno
	st.Comp = "ThermodynamicSimulation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["sim_id"]
	act.index["ThermodynamicSimulation_" + name] = st.Me;
	st.MyName = name
	act.ApThermodynamicSimulation = append(act.ApThermodynamicSimulation, st)
	return 0
}

func (me KpThermodynamicSimulation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu.unit:340, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApThermodynamicSimulation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ThermodynamicSimulation > tsu.unit:340, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ThermodynamicSimulation > tsu.unit:340, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpThermodynamicSimulation) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ThermodynamicSimulation %s,%s > tsu.unit:340, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpValidation struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Ktargetp int
	ItsPhysicalConstraint [] *KpPhysicalConstraint 
	Childs [] Kp
}

func (me KpValidation) TypeName() string {
    return me.Comp
}
func (me KpValidation) GetLineNo() string {
	return me.LineNo
}

func loadValidation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpValidation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApValidation)
	st.LineNo = lno
	st.Comp = "Validation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ktargetp = -1
	name,_ := st.Names["rule"]
	act.index["Validation_" + name] = st.Me;
	st.MyName = name
	act.ApValidation = append(act.ApValidation, st)
	return 0
}

func (me KpValidation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target" { // tsu.unit:356, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu.unit:351, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApValidation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Validation > tsu.unit:351, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Validation > tsu.unit:351, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpValidation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "PhysicalConstraint" { // tsu.unit:360, g_structh.act:676
		for _, st := range me.ItsPhysicalConstraint {
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
	if va[0] == "target" {
		if me.Ktargetp >= 0 {
			st := glob.Dats.ApModel[ me.Ktargetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Validation %s,%s > tsu.unit:351, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPhysicalConstraint struct {
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

func (me KpPhysicalConstraint) TypeName() string {
    return me.Comp
}
func (me KpPhysicalConstraint) GetLineNo() string {
	return me.LineNo
}

func loadPhysicalConstraint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPhysicalConstraint)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPhysicalConstraint)
	st.LineNo = lno
	st.Comp = "PhysicalConstraint";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApValidation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PhysicalConstraint has no Validation parent\n") ;
		return 1
	}
	st.Parent = act.ApValidation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PhysicalConstraint under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApValidation[ len( act.ApValidation )-1 ].Childs = append(act.ApValidation[ len( act.ApValidation )-1 ].Childs, st)
	act.ApValidation[ len( act.ApValidation )-1 ].ItsPhysicalConstraint = append(act.ApValidation[ len( act.ApValidation )-1 ].ItsPhysicalConstraint, st)	// tsu.unit:351, g_structh.act:403
	name,_ := st.Names["constraint"]
	s := strconv.Itoa(st.Kparentp) + "_PhysicalConstraint_" + name	// tsu.unit:364, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApPhysicalConstraint = append(act.ApPhysicalConstraint, st)
	return 0
}

func (me KpPhysicalConstraint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu.unit:351, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApValidation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:360, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPhysicalConstraint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PhysicalConstraint > tsu.unit:360, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PhysicalConstraint > tsu.unit:360, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPhysicalConstraint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:351, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApValidation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PhysicalConstraint %s,%s > tsu.unit:360, g_structh.act:209?", va[0], lno, me.LineNo)
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
	if (va[0] == "Tensor_distribution" && len(va) > 1) { // tsu.unit:183, g_structh.act:698
		for _, st := range glob.Dats.ApTensor {
			if (st.Kdistributionp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:375, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDistributionRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DistributionRule > tsu.unit:375, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DistributionRule > tsu.unit:375, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDistributionRule) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Tensor_distribution") { // tsu.unit:183, g_structh.act:583
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
	        fmt.Printf("?No its %s for DistributionRule %s,%s > tsu.unit:375, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSamplingAlgorithmRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpSamplingAlgorithmRule) TypeName() string {
    return me.Comp
}
func (me KpSamplingAlgorithmRule) GetLineNo() string {
	return me.LineNo
}

func loadSamplingAlgorithmRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSamplingAlgorithmRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSamplingAlgorithmRule)
	st.LineNo = lno
	st.Comp = "SamplingAlgorithmRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["algorithm"]
	act.index["SamplingAlgorithmRule_" + name] = st.Me;
	st.MyName = name
	act.ApSamplingAlgorithmRule = append(act.ApSamplingAlgorithmRule, st)
	return 0
}

func (me KpSamplingAlgorithmRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu.unit:386, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSamplingAlgorithmRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SamplingAlgorithmRule > tsu.unit:386, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SamplingAlgorithmRule > tsu.unit:386, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSamplingAlgorithmRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for SamplingAlgorithmRule %s,%s > tsu.unit:386, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBackendRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpBackendRule) TypeName() string {
    return me.Comp
}
func (me KpBackendRule) GetLineNo() string {
	return me.LineNo
}

func loadBackendRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpBackendRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBackendRule)
	st.LineNo = lno
	st.Comp = "BackendRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["backend"]
	act.index["BackendRule_" + name] = st.Me;
	st.MyName = name
	act.ApBackendRule = append(act.ApBackendRule, st)
	return 0
}

func (me KpBackendRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu.unit:396, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBackendRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,BackendRule > tsu.unit:396, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,BackendRule > tsu.unit:396, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBackendRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for BackendRule %s,%s > tsu.unit:396, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHardwareRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpHardwareRule) TypeName() string {
    return me.Comp
}
func (me KpHardwareRule) GetLineNo() string {
	return me.LineNo
}

func loadHardwareRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpHardwareRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHardwareRule)
	st.LineNo = lno
	st.Comp = "HardwareRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["hardware"]
	act.index["HardwareRule_" + name] = st.Me;
	st.MyName = name
	act.ApHardwareRule = append(act.ApHardwareRule, st)
	return 0
}

func (me KpHardwareRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu.unit:404, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHardwareRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,HardwareRule > tsu.unit:404, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,HardwareRule > tsu.unit:404, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHardwareRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for HardwareRule %s,%s > tsu.unit:404, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProject struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kdomainp int
	Kmodelp int
	ItsTargetConfig [] *KpTargetConfig 
	ItsBuildRule [] *KpBuildRule 
	Childs [] Kp
}

func (me KpProject) TypeName() string {
    return me.Comp
}
func (me KpProject) GetLineNo() string {
	return me.LineNo
}

func loadProject(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["project"]
	act.index["Project_" + name] = st.Me;
	st.MyName = name
	act.ApProject = append(act.ApProject, st)
	return 0
}

func (me KpProject) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "domain" { // tsu.unit:422, g_structh.act:609
		if (me.Kdomainp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Kdomainp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model" { // tsu.unit:423, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "BuildRule_project" && len(va) > 1) { // tsu.unit:446, g_structh.act:698
		for _, st := range glob.Dats.ApBuildRule {
			if (st.Kprojectp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu.unit:417, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Project > tsu.unit:417, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Project > tsu.unit:417, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProject) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TargetConfig" { // tsu.unit:426, g_structh.act:676
		for _, st := range me.ItsTargetConfig {
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
	if va[0] == "BuildRule" { // tsu.unit:438, g_structh.act:676
		for _, st := range me.ItsBuildRule {
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
	if (va[0] == "BuildRule_project") { // tsu.unit:446, g_structh.act:583
		for _, st := range glob.Dats.ApBuildRule {
			if (st.Kprojectp == me.Me) {
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
	        fmt.Printf("?No its %s for Project %s,%s > tsu.unit:417, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTargetConfig struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Khardwarep int
	Kconfigp int
}

func (me KpTargetConfig) TypeName() string {
    return me.Comp
}
func (me KpTargetConfig) GetLineNo() string {
	return me.LineNo
}

func loadTargetConfig(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTargetConfig)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTargetConfig)
	st.LineNo = lno
	st.Comp = "TargetConfig";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	st.Kconfigp = -1
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TargetConfig has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TargetConfig under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsTargetConfig = append(act.ApProject[ len( act.ApProject )-1 ].ItsTargetConfig, st)	// tsu.unit:417, g_structh.act:403
	name,_ := st.Names["target_id"]
	s := strconv.Itoa(st.Kparentp) + "_TargetConfig_" + name	// tsu.unit:430, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTargetConfig = append(act.ApTargetConfig, st)
	return 0
}

func (me KpTargetConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // tsu.unit:431, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "config" { // tsu.unit:433, g_structh.act:609
		if (me.Kconfigp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kconfigp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:417, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:426, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTargetConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TargetConfig > tsu.unit:426, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TargetConfig > tsu.unit:426, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTargetConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:417, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
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
	if (va[0] == "BuildRule_validate_against") { // tsu.unit:447, g_structh.act:583
		for _, st := range glob.Dats.ApBuildRule {
			if (st.Kvalidate_againstp == me.Me) {
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
	        fmt.Printf("?No its %s for TargetConfig %s,%s > tsu.unit:426, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpBuildRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kprojectp int
	Kvalidate_againstp int
}

func (me KpBuildRule) TypeName() string {
    return me.Comp
}
func (me KpBuildRule) GetLineNo() string {
	return me.LineNo
}

func loadBuildRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpBuildRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApBuildRule)
	st.LineNo = lno
	st.Comp = "BuildRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kprojectp = -1
	st.Kvalidate_againstp = -1
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " BuildRule has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " BuildRule under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsBuildRule = append(act.ApProject[ len( act.ApProject )-1 ].ItsBuildRule, st)	// tsu.unit:417, g_structh.act:403
	name,_ := st.Names["build_id"]
	s := strconv.Itoa(st.Kparentp) + "_BuildRule_" + name	// tsu.unit:442, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBuildRule = append(act.ApBuildRule, st)
	return 0
}

func (me KpBuildRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project" { // tsu.unit:446, g_structh.act:609
		if (me.Kprojectp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kprojectp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "validate_against" { // tsu.unit:447, g_structh.act:609
		if (me.Kvalidate_againstp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTargetConfig[ me.Kvalidate_againstp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:417, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu.unit:438, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBuildRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,BuildRule > tsu.unit:438, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,BuildRule > tsu.unit:438, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBuildRule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:417, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "project" {
		if me.Kprojectp >= 0 {
			st := glob.Dats.ApProject[ me.Kprojectp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "validate_against" {
		if me.Kvalidate_againstp >= 0 {
			st := glob.Dats.ApTargetConfig[ me.Kvalidate_againstp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for BuildRule %s,%s > tsu.unit:438, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Ktargetp int
	ItsSearchParameter [] *KpSearchParameter 
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
	st.Ktargetp = -1
	name,_ := st.Names["search_space"]
	act.index["SearchSpace_" + name] = st.Me;
	st.MyName = name
	act.ApSearchSpace = append(act.ApSearchSpace, st)
	return 0
}

func (me KpSearchSpace) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target" { // tsu-auto.unit:10, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "EvolutionStrategy_search_space" && len(va) > 1) { // tsu-auto.unit:47, g_structh.act:698
		for _, st := range glob.Dats.ApEvolutionStrategy {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SensitivityAnalysis_search_space" && len(va) > 1) { // tsu-auto.unit:176, g_structh.act:698
		for _, st := range glob.Dats.ApSensitivityAnalysis {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:5, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpace > tsu-auto.unit:5, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchSpace > tsu-auto.unit:5, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchSpace) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "SearchParameter" { // tsu-auto.unit:15, g_structh.act:676
		for _, st := range me.ItsSearchParameter {
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
	if va[0] == "target" {
		if me.Ktargetp >= 0 {
			st := glob.Dats.ApConfig[ me.Ktargetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "EvolutionStrategy_search_space") { // tsu-auto.unit:47, g_structh.act:583
		for _, st := range glob.Dats.ApEvolutionStrategy {
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
	if (va[0] == "SensitivityAnalysis_search_space") { // tsu-auto.unit:176, g_structh.act:583
		for _, st := range glob.Dats.ApSensitivityAnalysis {
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
	        fmt.Printf("?No its %s for SearchSpace %s,%s > tsu-auto.unit:5, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchParameter struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsScheduleParameter [] *KpScheduleParameter 
	Childs [] Kp
}

func (me KpSearchParameter) TypeName() string {
    return me.Comp
}
func (me KpSearchParameter) GetLineNo() string {
	return me.LineNo
}

func loadSearchParameter(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchParameter)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchParameter)
	st.LineNo = lno
	st.Comp = "SearchParameter";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchParameter has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchParameter under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchParameter = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchParameter, st)	// tsu-auto.unit:5, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_SearchParameter_" + name	// tsu-auto.unit:19, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchParameter = append(act.ApSearchParameter, st)
	return 0
}

func (me KpSearchParameter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:5, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:15, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchParameter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchParameter > tsu-auto.unit:15, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchParameter > tsu-auto.unit:15, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchParameter) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ScheduleParameter" { // tsu-auto.unit:29, g_structh.act:676
		for _, st := range me.ItsScheduleParameter {
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
	if va[0] == "parent" { // tsu-auto.unit:5, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchParameter %s,%s > tsu-auto.unit:15, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpScheduleParameter struct {
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

func (me KpScheduleParameter) TypeName() string {
    return me.Comp
}
func (me KpScheduleParameter) GetLineNo() string {
	return me.LineNo
}

func loadScheduleParameter(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpScheduleParameter)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApScheduleParameter)
	st.LineNo = lno
	st.Comp = "ScheduleParameter";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSearchParameter ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ScheduleParameter has no SearchParameter parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchParameter[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ScheduleParameter under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchParameter[ len( act.ApSearchParameter )-1 ].Childs = append(act.ApSearchParameter[ len( act.ApSearchParameter )-1 ].Childs, st)
	act.ApSearchParameter[ len( act.ApSearchParameter )-1 ].ItsScheduleParameter = append(act.ApSearchParameter[ len( act.ApSearchParameter )-1 ].ItsScheduleParameter, st)	// tsu-auto.unit:15, g_structh.act:403
	act.ApScheduleParameter = append(act.ApScheduleParameter, st)
	return 0
}

func (me KpScheduleParameter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:15, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchParameter[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:29, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApScheduleParameter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ScheduleParameter > tsu-auto.unit:29, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ScheduleParameter > tsu-auto.unit:29, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpScheduleParameter) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:15, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchParameter[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ScheduleParameter %s,%s > tsu-auto.unit:29, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEvolutionStrategy struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Ksearch_spacep int
	ItsFitnessFunction [] *KpFitnessFunction 
	ItsMultiObjective [] *KpMultiObjective 
	Childs [] Kp
}

func (me KpEvolutionStrategy) TypeName() string {
    return me.Comp
}
func (me KpEvolutionStrategy) GetLineNo() string {
	return me.LineNo
}

func loadEvolutionStrategy(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEvolutionStrategy)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEvolutionStrategy)
	st.LineNo = lno
	st.Comp = "EvolutionStrategy";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ksearch_spacep = -1
	name,_ := st.Names["strategy"]
	act.index["EvolutionStrategy_" + name] = st.Me;
	st.MyName = name
	act.ApEvolutionStrategy = append(act.ApEvolutionStrategy, st)
	return 0
}

func (me KpEvolutionStrategy) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "search_space" { // tsu-auto.unit:47, g_structh.act:609
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "ExperimentTrial_strategy" && len(va) > 1) { // tsu-auto.unit:96, g_structh.act:698
		for _, st := range glob.Dats.ApExperimentTrial {
			if (st.Kstrategyp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OptimizationRun_strategy" && len(va) > 1) { // tsu-auto.unit:118, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kstrategyp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "AutoTuneConfig_strategy" && len(va) > 1) { // tsu-auto.unit:165, g_structh.act:698
		for _, st := range glob.Dats.ApAutoTuneConfig {
			if (st.Kstrategyp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:41, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEvolutionStrategy[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EvolutionStrategy > tsu-auto.unit:41, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EvolutionStrategy > tsu-auto.unit:41, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEvolutionStrategy) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "FitnessFunction" { // tsu-auto.unit:57, g_structh.act:676
		for _, st := range me.ItsFitnessFunction {
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
	if va[0] == "MultiObjective" { // tsu-auto.unit:224, g_structh.act:676
		for _, st := range me.ItsMultiObjective {
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
	if (va[0] == "ExperimentTrial_strategy") { // tsu-auto.unit:96, g_structh.act:583
		for _, st := range glob.Dats.ApExperimentTrial {
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
	if (va[0] == "OptimizationRun_strategy") { // tsu-auto.unit:118, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationRun {
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
	if (va[0] == "AutoTuneConfig_strategy") { // tsu-auto.unit:165, g_structh.act:583
		for _, st := range glob.Dats.ApAutoTuneConfig {
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
	        fmt.Printf("?No its %s for EvolutionStrategy %s,%s > tsu-auto.unit:41, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFitnessFunction struct {
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

func (me KpFitnessFunction) TypeName() string {
    return me.Comp
}
func (me KpFitnessFunction) GetLineNo() string {
	return me.LineNo
}

func loadFitnessFunction(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFitnessFunction)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFitnessFunction)
	st.LineNo = lno
	st.Comp = "FitnessFunction";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApEvolutionStrategy ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FitnessFunction has no EvolutionStrategy parent\n") ;
		return 1
	}
	st.Parent = act.ApEvolutionStrategy[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FitnessFunction under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].Childs = append(act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].Childs, st)
	act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].ItsFitnessFunction = append(act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].ItsFitnessFunction, st)	// tsu-auto.unit:41, g_structh.act:403
	name,_ := st.Names["fitness_fn"]
	s := strconv.Itoa(st.Kparentp) + "_FitnessFunction_" + name	// tsu-auto.unit:61, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApFitnessFunction = append(act.ApFitnessFunction, st)
	return 0
}

func (me KpFitnessFunction) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:41, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEvolutionStrategy[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:57, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFitnessFunction[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FitnessFunction > tsu-auto.unit:57, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FitnessFunction > tsu-auto.unit:57, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFitnessFunction) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:41, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApEvolutionStrategy[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for FitnessFunction %s,%s > tsu-auto.unit:57, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAdaptiveParameter struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpAdaptiveParameter) TypeName() string {
    return me.Comp
}
func (me KpAdaptiveParameter) GetLineNo() string {
	return me.LineNo
}

func loadAdaptiveParameter(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAdaptiveParameter)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAdaptiveParameter)
	st.LineNo = lno
	st.Comp = "AdaptiveParameter";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["param"]
	act.index["AdaptiveParameter_" + name] = st.Me;
	st.MyName = name
	act.ApAdaptiveParameter = append(act.ApAdaptiveParameter, st)
	return 0
}

func (me KpAdaptiveParameter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:68, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAdaptiveParameter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AdaptiveParameter > tsu-auto.unit:68, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AdaptiveParameter > tsu-auto.unit:68, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAdaptiveParameter) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for AdaptiveParameter %s,%s > tsu-auto.unit:68, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPerformanceMetric struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpPerformanceMetric) TypeName() string {
    return me.Comp
}
func (me KpPerformanceMetric) GetLineNo() string {
	return me.LineNo
}

func loadPerformanceMetric(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPerformanceMetric)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPerformanceMetric)
	st.LineNo = lno
	st.Comp = "PerformanceMetric";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["metric"]
	act.index["PerformanceMetric_" + name] = st.Me;
	st.MyName = name
	act.ApPerformanceMetric = append(act.ApPerformanceMetric, st)
	return 0
}

func (me KpPerformanceMetric) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "TrialMetrics_metric" && len(va) > 1) { // tsu-auto.unit:108, g_structh.act:698
		for _, st := range glob.Dats.ApTrialMetrics {
			if (st.Kmetricp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:80, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPerformanceMetric[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PerformanceMetric > tsu-auto.unit:80, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PerformanceMetric > tsu-auto.unit:80, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPerformanceMetric) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "TrialMetrics_metric") { // tsu-auto.unit:108, g_structh.act:583
		for _, st := range glob.Dats.ApTrialMetrics {
			if (st.Kmetricp == me.Me) {
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
	        fmt.Printf("?No its %s for PerformanceMetric %s,%s > tsu-auto.unit:80, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpExperimentTrial struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kstrategyp int
	ItsTrialMetrics [] *KpTrialMetrics 
	Childs [] Kp
}

func (me KpExperimentTrial) TypeName() string {
    return me.Comp
}
func (me KpExperimentTrial) GetLineNo() string {
	return me.LineNo
}

func loadExperimentTrial(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpExperimentTrial)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApExperimentTrial)
	st.LineNo = lno
	st.Comp = "ExperimentTrial";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kstrategyp = -1
	name,_ := st.Names["trial_id"]
	act.index["ExperimentTrial_" + name] = st.Me;
	st.MyName = name
	act.ApExperimentTrial = append(act.ApExperimentTrial, st)
	return 0
}

func (me KpExperimentTrial) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "strategy" { // tsu-auto.unit:96, g_structh.act:609
		if (me.Kstrategyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEvolutionStrategy[ me.Kstrategyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "OptimizationRun_best_trial" && len(va) > 1) { // tsu-auto.unit:122, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kbest_trialp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:91, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExperimentTrial[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ExperimentTrial > tsu-auto.unit:91, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ExperimentTrial > tsu-auto.unit:91, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExperimentTrial) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TrialMetrics" { // tsu-auto.unit:104, g_structh.act:676
		for _, st := range me.ItsTrialMetrics {
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
	if va[0] == "strategy" {
		if me.Kstrategyp >= 0 {
			st := glob.Dats.ApEvolutionStrategy[ me.Kstrategyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "OptimizationRun_best_trial") { // tsu-auto.unit:122, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kbest_trialp == me.Me) {
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
	        fmt.Printf("?No its %s for ExperimentTrial %s,%s > tsu-auto.unit:91, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTrialMetrics struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmetricp int
}

func (me KpTrialMetrics) TypeName() string {
    return me.Comp
}
func (me KpTrialMetrics) GetLineNo() string {
	return me.LineNo
}

func loadTrialMetrics(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTrialMetrics)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTrialMetrics)
	st.LineNo = lno
	st.Comp = "TrialMetrics";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmetricp = -1
	st.Kparentp = len( act.ApExperimentTrial ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TrialMetrics has no ExperimentTrial parent\n") ;
		return 1
	}
	st.Parent = act.ApExperimentTrial[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TrialMetrics under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApExperimentTrial[ len( act.ApExperimentTrial )-1 ].Childs = append(act.ApExperimentTrial[ len( act.ApExperimentTrial )-1 ].Childs, st)
	act.ApExperimentTrial[ len( act.ApExperimentTrial )-1 ].ItsTrialMetrics = append(act.ApExperimentTrial[ len( act.ApExperimentTrial )-1 ].ItsTrialMetrics, st)	// tsu-auto.unit:91, g_structh.act:403
	act.ApTrialMetrics = append(act.ApTrialMetrics, st)
	return 0
}

func (me KpTrialMetrics) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "metric" { // tsu-auto.unit:108, g_structh.act:609
		if (me.Kmetricp >= 0 && len(va) > 1) {
			return( glob.Dats.ApPerformanceMetric[ me.Kmetricp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu-auto.unit:91, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExperimentTrial[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:104, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTrialMetrics[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TrialMetrics > tsu-auto.unit:104, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TrialMetrics > tsu-auto.unit:104, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTrialMetrics) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:91, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApExperimentTrial[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "metric" {
		if me.Kmetricp >= 0 {
			st := glob.Dats.ApPerformanceMetric[ me.Kmetricp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TrialMetrics %s,%s > tsu-auto.unit:104, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOptimizationRun struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kstrategyp int
	Kbest_trialp int
	ItsCheckpoint [] *KpCheckpoint 
	ItsParetoFront [] *KpParetoFront 
	ItsResourceBudget [] *KpResourceBudget 
	Childs [] Kp
}

func (me KpOptimizationRun) TypeName() string {
    return me.Comp
}
func (me KpOptimizationRun) GetLineNo() string {
	return me.LineNo
}

func loadOptimizationRun(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOptimizationRun)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOptimizationRun)
	st.LineNo = lno
	st.Comp = "OptimizationRun";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kstrategyp = -1
	st.Kbest_trialp = -1
	name,_ := st.Names["run_id"]
	act.index["OptimizationRun_" + name] = st.Me;
	st.MyName = name
	act.ApOptimizationRun = append(act.ApOptimizationRun, st)
	return 0
}

func (me KpOptimizationRun) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "strategy" { // tsu-auto.unit:118, g_structh.act:609
		if (me.Kstrategyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEvolutionStrategy[ me.Kstrategyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "best_trial" { // tsu-auto.unit:122, g_structh.act:609
		if (me.Kbest_trialp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExperimentTrial[ me.Kbest_trialp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "HyperParameterRule_learned_from" && len(va) > 1) { // tsu-auto.unit:145, g_structh.act:698
		for _, st := range glob.Dats.ApHyperParameterRule {
			if (st.Klearned_fromp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OptimizationPattern_discovered_by" && len(va) > 1) { // tsu-auto.unit:206, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationPattern {
			if (st.Kdiscovered_byp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:113, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OptimizationRun > tsu-auto.unit:113, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OptimizationRun > tsu-auto.unit:113, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOptimizationRun) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Checkpoint" { // tsu-auto.unit:126, g_structh.act:676
		for _, st := range me.ItsCheckpoint {
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
	if va[0] == "ParetoFront" { // tsu-auto.unit:234, g_structh.act:676
		for _, st := range me.ItsParetoFront {
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
	if va[0] == "ResourceBudget" { // tsu-auto.unit:275, g_structh.act:676
		for _, st := range me.ItsResourceBudget {
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
	if va[0] == "strategy" {
		if me.Kstrategyp >= 0 {
			st := glob.Dats.ApEvolutionStrategy[ me.Kstrategyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "best_trial" {
		if me.Kbest_trialp >= 0 {
			st := glob.Dats.ApExperimentTrial[ me.Kbest_trialp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "HyperParameterRule_learned_from") { // tsu-auto.unit:145, g_structh.act:583
		for _, st := range glob.Dats.ApHyperParameterRule {
			if (st.Klearned_fromp == me.Me) {
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
	if (va[0] == "OptimizationPattern_discovered_by") { // tsu-auto.unit:206, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationPattern {
			if (st.Kdiscovered_byp == me.Me) {
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
	        fmt.Printf("?No its %s for OptimizationRun %s,%s > tsu-auto.unit:113, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
	Kparentp int
}

func (me KpCheckpoint) TypeName() string {
    return me.Comp
}
func (me KpCheckpoint) GetLineNo() string {
	return me.LineNo
}

func loadCheckpoint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	st.Kparentp = len( act.ApOptimizationRun ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Checkpoint has no OptimizationRun parent\n") ;
		return 1
	}
	st.Parent = act.ApOptimizationRun[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Checkpoint under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].Childs = append(act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].Childs, st)
	act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].ItsCheckpoint = append(act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].ItsCheckpoint, st)	// tsu-auto.unit:113, g_structh.act:403
	name,_ := st.Names["checkpoint_id"]
	s := strconv.Itoa(st.Kparentp) + "_Checkpoint_" + name	// tsu-auto.unit:130, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCheckpoint = append(act.ApCheckpoint, st)
	return 0
}

func (me KpCheckpoint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:113, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "AutoTuneConfig_resume_from" && len(va) > 1) { // tsu-auto.unit:168, g_structh.act:698
		for _, st := range glob.Dats.ApAutoTuneConfig {
			if (st.Kresume_fromp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:126, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCheckpoint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Checkpoint > tsu-auto.unit:126, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Checkpoint > tsu-auto.unit:126, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCheckpoint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:113, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOptimizationRun[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "AutoTuneConfig_resume_from") { // tsu-auto.unit:168, g_structh.act:583
		for _, st := range glob.Dats.ApAutoTuneConfig {
			if (st.Kresume_fromp == me.Me) {
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
	        fmt.Printf("?No its %s for Checkpoint %s,%s > tsu-auto.unit:126, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHyperParameterRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Klearned_fromp int
}

func (me KpHyperParameterRule) TypeName() string {
    return me.Comp
}
func (me KpHyperParameterRule) GetLineNo() string {
	return me.LineNo
}

func loadHyperParameterRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpHyperParameterRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHyperParameterRule)
	st.LineNo = lno
	st.Comp = "HyperParameterRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Klearned_fromp = -1
	name,_ := st.Names["rule_id"]
	act.index["HyperParameterRule_" + name] = st.Me;
	st.MyName = name
	act.ApHyperParameterRule = append(act.ApHyperParameterRule, st)
	return 0
}

func (me KpHyperParameterRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "learned_from" { // tsu-auto.unit:145, g_structh.act:609
		if (me.Klearned_fromp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Klearned_fromp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:137, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHyperParameterRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,HyperParameterRule > tsu-auto.unit:137, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,HyperParameterRule > tsu-auto.unit:137, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHyperParameterRule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "learned_from" {
		if me.Klearned_fromp >= 0 {
			st := glob.Dats.ApOptimizationRun[ me.Klearned_fromp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for HyperParameterRule %s,%s > tsu-auto.unit:137, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMetaLearning struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpMetaLearning) TypeName() string {
    return me.Comp
}
func (me KpMetaLearning) GetLineNo() string {
	return me.LineNo
}

func loadMetaLearning(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMetaLearning)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMetaLearning)
	st.LineNo = lno
	st.Comp = "MetaLearning";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["meta_id"]
	act.index["MetaLearning_" + name] = st.Me;
	st.MyName = name
	act.ApMetaLearning = append(act.ApMetaLearning, st)
	return 0
}

func (me KpMetaLearning) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:148, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMetaLearning[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MetaLearning > tsu-auto.unit:148, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MetaLearning > tsu-auto.unit:148, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMetaLearning) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for MetaLearning %s,%s > tsu-auto.unit:148, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAutoTuneConfig struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kstrategyp int
	Kresume_fromp int
}

func (me KpAutoTuneConfig) TypeName() string {
    return me.Comp
}
func (me KpAutoTuneConfig) GetLineNo() string {
	return me.LineNo
}

func loadAutoTuneConfig(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAutoTuneConfig)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAutoTuneConfig)
	st.LineNo = lno
	st.Comp = "AutoTuneConfig";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kstrategyp = -1
	st.Kresume_fromp = -1
	st.Kparentp = len( act.ApConfig ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AutoTuneConfig has no Config parent\n") ;
		return 1
	}
	st.Parent = act.ApConfig[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AutoTuneConfig under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApConfig[ len( act.ApConfig )-1 ].Childs = append(act.ApConfig[ len( act.ApConfig )-1 ].Childs, st)
	act.ApConfig[ len( act.ApConfig )-1 ].ItsAutoTuneConfig = append(act.ApConfig[ len( act.ApConfig )-1 ].ItsAutoTuneConfig, st)	// tsu.unit:317, g_structh.act:403
	name,_ := st.Names["autotune"]
	s := strconv.Itoa(st.Kparentp) + "_AutoTuneConfig_" + name	// tsu-auto.unit:163, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApAutoTuneConfig = append(act.ApAutoTuneConfig, st)
	return 0
}

func (me KpAutoTuneConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "strategy" { // tsu-auto.unit:165, g_structh.act:609
		if (me.Kstrategyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEvolutionStrategy[ me.Kstrategyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "resume_from" { // tsu-auto.unit:168, g_structh.act:609
		if (me.Kresume_fromp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCheckpoint[ me.Kresume_fromp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // tsu.unit:317, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:159, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAutoTuneConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AutoTuneConfig > tsu-auto.unit:159, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AutoTuneConfig > tsu-auto.unit:159, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAutoTuneConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu.unit:317, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApConfig[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "strategy" {
		if me.Kstrategyp >= 0 {
			st := glob.Dats.ApEvolutionStrategy[ me.Kstrategyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "resume_from" {
		if me.Kresume_fromp >= 0 {
			st := glob.Dats.ApCheckpoint[ me.Kresume_fromp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for AutoTuneConfig %s,%s > tsu-auto.unit:159, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSensitivityAnalysis struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Ksearch_spacep int
}

func (me KpSensitivityAnalysis) TypeName() string {
    return me.Comp
}
func (me KpSensitivityAnalysis) GetLineNo() string {
	return me.LineNo
}

func loadSensitivityAnalysis(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSensitivityAnalysis)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSensitivityAnalysis)
	st.LineNo = lno
	st.Comp = "SensitivityAnalysis";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ksearch_spacep = -1
	name,_ := st.Names["analysis_id"]
	act.index["SensitivityAnalysis_" + name] = st.Me;
	st.MyName = name
	act.ApSensitivityAnalysis = append(act.ApSensitivityAnalysis, st)
	return 0
}

func (me KpSensitivityAnalysis) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "search_space" { // tsu-auto.unit:176, g_structh.act:609
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:171, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSensitivityAnalysis[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SensitivityAnalysis > tsu-auto.unit:171, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SensitivityAnalysis > tsu-auto.unit:171, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSensitivityAnalysis) DoIts(glob *GlobT, va []string, lno string) int {
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
	        fmt.Printf("?No its %s for SensitivityAnalysis %s,%s > tsu-auto.unit:171, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnsembleStrategy struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpEnsembleStrategy) TypeName() string {
    return me.Comp
}
func (me KpEnsembleStrategy) GetLineNo() string {
	return me.LineNo
}

func loadEnsembleStrategy(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEnsembleStrategy)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnsembleStrategy)
	st.LineNo = lno
	st.Comp = "EnsembleStrategy";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["ensemble_id"]
	act.index["EnsembleStrategy_" + name] = st.Me;
	st.MyName = name
	act.ApEnsembleStrategy = append(act.ApEnsembleStrategy, st)
	return 0
}

func (me KpEnsembleStrategy) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:182, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnsembleStrategy[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnsembleStrategy > tsu-auto.unit:182, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnsembleStrategy > tsu-auto.unit:182, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnsembleStrategy) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for EnsembleStrategy %s,%s > tsu-auto.unit:182, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOptimizationPattern struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kdiscovered_byp int
}

func (me KpOptimizationPattern) TypeName() string {
    return me.Comp
}
func (me KpOptimizationPattern) GetLineNo() string {
	return me.LineNo
}

func loadOptimizationPattern(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOptimizationPattern)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOptimizationPattern)
	st.LineNo = lno
	st.Comp = "OptimizationPattern";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kdiscovered_byp = -1
	name,_ := st.Names["pattern_id"]
	act.index["OptimizationPattern_" + name] = st.Me;
	st.MyName = name
	act.ApOptimizationPattern = append(act.ApOptimizationPattern, st)
	return 0
}

func (me KpOptimizationPattern) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "discovered_by" { // tsu-auto.unit:206, g_structh.act:609
		if (me.Kdiscovered_byp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Kdiscovered_byp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:196, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationPattern[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OptimizationPattern > tsu-auto.unit:196, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OptimizationPattern > tsu-auto.unit:196, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOptimizationPattern) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "discovered_by" {
		if me.Kdiscovered_byp >= 0 {
			st := glob.Dats.ApOptimizationRun[ me.Kdiscovered_byp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for OptimizationPattern %s,%s > tsu-auto.unit:196, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTransferLearning struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpTransferLearning) TypeName() string {
    return me.Comp
}
func (me KpTransferLearning) GetLineNo() string {
	return me.LineNo
}

func loadTransferLearning(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTransferLearning)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTransferLearning)
	st.LineNo = lno
	st.Comp = "TransferLearning";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["transfer_id"]
	act.index["TransferLearning_" + name] = st.Me;
	st.MyName = name
	act.ApTransferLearning = append(act.ApTransferLearning, st)
	return 0
}

func (me KpTransferLearning) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:209, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTransferLearning[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TransferLearning > tsu-auto.unit:209, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TransferLearning > tsu-auto.unit:209, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTransferLearning) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for TransferLearning %s,%s > tsu-auto.unit:209, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMultiObjective struct {
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

func (me KpMultiObjective) TypeName() string {
    return me.Comp
}
func (me KpMultiObjective) GetLineNo() string {
	return me.LineNo
}

func loadMultiObjective(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMultiObjective)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMultiObjective)
	st.LineNo = lno
	st.Comp = "MultiObjective";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApEvolutionStrategy ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MultiObjective has no EvolutionStrategy parent\n") ;
		return 1
	}
	st.Parent = act.ApEvolutionStrategy[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MultiObjective under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].Childs = append(act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].Childs, st)
	act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].ItsMultiObjective = append(act.ApEvolutionStrategy[ len( act.ApEvolutionStrategy )-1 ].ItsMultiObjective, st)	// tsu-auto.unit:41, g_structh.act:403
	act.ApMultiObjective = append(act.ApMultiObjective, st)
	return 0
}

func (me KpMultiObjective) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:41, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEvolutionStrategy[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:224, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMultiObjective[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MultiObjective > tsu-auto.unit:224, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MultiObjective > tsu-auto.unit:224, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMultiObjective) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:41, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApEvolutionStrategy[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MultiObjective %s,%s > tsu-auto.unit:224, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpParetoFront struct {
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

func (me KpParetoFront) TypeName() string {
    return me.Comp
}
func (me KpParetoFront) GetLineNo() string {
	return me.LineNo
}

func loadParetoFront(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpParetoFront)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApParetoFront)
	st.LineNo = lno
	st.Comp = "ParetoFront";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOptimizationRun ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ParetoFront has no OptimizationRun parent\n") ;
		return 1
	}
	st.Parent = act.ApOptimizationRun[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ParetoFront under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].Childs = append(act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].Childs, st)
	act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].ItsParetoFront = append(act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].ItsParetoFront, st)	// tsu-auto.unit:113, g_structh.act:403
	name,_ := st.Names["front_id"]
	s := strconv.Itoa(st.Kparentp) + "_ParetoFront_" + name	// tsu-auto.unit:238, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApParetoFront = append(act.ApParetoFront, st)
	return 0
}

func (me KpParetoFront) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:113, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:234, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApParetoFront[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ParetoFront > tsu-auto.unit:234, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ParetoFront > tsu-auto.unit:234, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpParetoFront) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:113, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOptimizationRun[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ParetoFront %s,%s > tsu-auto.unit:234, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDiagnosticRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDiagnosticRule) TypeName() string {
    return me.Comp
}
func (me KpDiagnosticRule) GetLineNo() string {
	return me.LineNo
}

func loadDiagnosticRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDiagnosticRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDiagnosticRule)
	st.LineNo = lno
	st.Comp = "DiagnosticRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["rule_id"]
	act.index["DiagnosticRule_" + name] = st.Me;
	st.MyName = name
	act.ApDiagnosticRule = append(act.ApDiagnosticRule, st)
	return 0
}

func (me KpDiagnosticRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:248, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDiagnosticRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DiagnosticRule > tsu-auto.unit:248, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DiagnosticRule > tsu-auto.unit:248, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDiagnosticRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DiagnosticRule %s,%s > tsu-auto.unit:248, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFailureMode struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpFailureMode) TypeName() string {
    return me.Comp
}
func (me KpFailureMode) GetLineNo() string {
	return me.LineNo
}

func loadFailureMode(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFailureMode)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFailureMode)
	st.LineNo = lno
	st.Comp = "FailureMode";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["failure_id"]
	act.index["FailureMode_" + name] = st.Me;
	st.MyName = name
	act.ApFailureMode = append(act.ApFailureMode, st)
	return 0
}

func (me KpFailureMode) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:259, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFailureMode[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FailureMode > tsu-auto.unit:259, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FailureMode > tsu-auto.unit:259, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFailureMode) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for FailureMode %s,%s > tsu-auto.unit:259, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpResourceBudget struct {
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

func (me KpResourceBudget) TypeName() string {
    return me.Comp
}
func (me KpResourceBudget) GetLineNo() string {
	return me.LineNo
}

func loadResourceBudget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpResourceBudget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApResourceBudget)
	st.LineNo = lno
	st.Comp = "ResourceBudget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOptimizationRun ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ResourceBudget has no OptimizationRun parent\n") ;
		return 1
	}
	st.Parent = act.ApOptimizationRun[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ResourceBudget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].Childs = append(act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].Childs, st)
	act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].ItsResourceBudget = append(act.ApOptimizationRun[ len( act.ApOptimizationRun )-1 ].ItsResourceBudget, st)	// tsu-auto.unit:113, g_structh.act:403
	name,_ := st.Names["budget_id"]
	s := strconv.Itoa(st.Kparentp) + "_ResourceBudget_" + name	// tsu-auto.unit:279, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApResourceBudget = append(act.ApResourceBudget, st)
	return 0
}

func (me KpResourceBudget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // tsu-auto.unit:113, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // tsu-auto.unit:275, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApResourceBudget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ResourceBudget > tsu-auto.unit:275, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ResourceBudget > tsu-auto.unit:275, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpResourceBudget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // tsu-auto.unit:113, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOptimizationRun[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ResourceBudget %s,%s > tsu-auto.unit:275, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAdaptiveScheduling struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpAdaptiveScheduling) TypeName() string {
    return me.Comp
}
func (me KpAdaptiveScheduling) GetLineNo() string {
	return me.LineNo
}

func loadAdaptiveScheduling(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAdaptiveScheduling)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAdaptiveScheduling)
	st.LineNo = lno
	st.Comp = "AdaptiveScheduling";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["schedule_id"]
	act.index["AdaptiveScheduling_" + name] = st.Me;
	st.MyName = name
	act.ApAdaptiveScheduling = append(act.ApAdaptiveScheduling, st)
	return 0
}

func (me KpAdaptiveScheduling) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // tsu-auto.unit:287, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAdaptiveScheduling[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AdaptiveScheduling > tsu-auto.unit:287, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AdaptiveScheduling > tsu-auto.unit:287, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAdaptiveScheduling) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for AdaptiveScheduling %s,%s > tsu-auto.unit:287, g_structh.act:209?", va[0], lno, me.LineNo)
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

