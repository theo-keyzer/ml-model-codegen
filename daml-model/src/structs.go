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
	if (va[0] == "Project_domain" && len(va) > 1) { // daml.unit:628, g_structh.act:698
		for _, st := range glob.Dats.ApProject {
			if (st.Kdomainp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:11, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Domain > daml.unit:11, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Domain > daml.unit:11, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDomain) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Project_domain") { // daml.unit:628, g_structh.act:583
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
	        fmt.Printf("?No its %s for Domain %s,%s > daml.unit:11, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsGPU [] *KpGPU 
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
	if (va[0] == "Kernel_hardware" && len(va) > 1) { // daml.unit:51, g_structh.act:698
		for _, st := range glob.Dats.ApKernel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "DynamicKernel_hardware" && len(va) > 1) { // daml.unit:64, g_structh.act:698
		for _, st := range glob.Dats.ApDynamicKernel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Optimization_target" && len(va) > 1) { // daml.unit:98, g_structh.act:698
		for _, st := range glob.Dats.ApOptimization {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Fusion_hardware" && len(va) > 1) { // daml.unit:110, g_structh.act:698
		for _, st := range glob.Dats.ApFusion {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchOp_hardware" && len(va) > 1) { // daml.unit:190, g_structh.act:698
		for _, st := range glob.Dats.ApSearchOp {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ExpertRoutingOp_hardware" && len(va) > 1) { // daml.unit:249, g_structh.act:698
		for _, st := range glob.Dats.ApExpertRoutingOp {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Model_hardware" && len(va) > 1) { // daml.unit:457, g_structh.act:698
		for _, st := range glob.Dats.ApModel {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Config_target" && len(va) > 1) { // daml.unit:548, g_structh.act:698
		for _, st := range glob.Dats.ApConfig {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "TargetConfig_hardware" && len(va) > 1) { // daml.unit:637, g_structh.act:698
		for _, st := range glob.Dats.ApTargetConfig {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:19, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Hardware > daml.unit:19, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Hardware > daml.unit:19, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "GPU" { // daml.unit:31, g_structh.act:676
		for _, st := range me.ItsGPU {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "Kernel_hardware") { // daml.unit:51, g_structh.act:583
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
	if (va[0] == "DynamicKernel_hardware") { // daml.unit:64, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicKernel {
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
	if (va[0] == "Optimization_target") { // daml.unit:98, g_structh.act:583
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
	if (va[0] == "Fusion_hardware") { // daml.unit:110, g_structh.act:583
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
	if (va[0] == "SearchOp_hardware") { // daml.unit:190, g_structh.act:583
		for _, st := range glob.Dats.ApSearchOp {
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
	if (va[0] == "ExpertRoutingOp_hardware") { // daml.unit:249, g_structh.act:583
		for _, st := range glob.Dats.ApExpertRoutingOp {
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
	if (va[0] == "Model_hardware") { // daml.unit:457, g_structh.act:583
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
	if (va[0] == "Config_target") { // daml.unit:548, g_structh.act:583
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
	if (va[0] == "TargetConfig_hardware") { // daml.unit:637, g_structh.act:583
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
	        fmt.Printf("?No its %s for Hardware %s,%s > daml.unit:19, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpGPU struct {
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

func (me KpGPU) TypeName() string {
    return me.Comp
}
func (me KpGPU) GetLineNo() string {
	return me.LineNo
}

func loadGPU(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpGPU)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApGPU)
	st.LineNo = lno
	st.Comp = "GPU";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " GPU has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " GPU under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsGPU = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsGPU, st)	// daml.unit:19, g_structh.act:403
	name,_ := st.Names["gpu_id"]
	s := strconv.Itoa(st.Kparentp) + "_GPU_" + name	// daml.unit:35, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApGPU = append(act.ApGPU, st)
	return 0
}

func (me KpGPU) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:19, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:31, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApGPU[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,GPU > daml.unit:31, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,GPU > daml.unit:31, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpGPU) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:19, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "DynamicKernel_gpu_target") { // daml.unit:65, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicKernel {
			if (st.Kgpu_targetp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for GPU %s,%s > daml.unit:31, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsDynamicKernel [] *KpDynamicKernel 
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
	if va[0] == "hardware" { // daml.unit:51, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "DynamicKernel_copy_kernel" && len(va) > 1) { // daml.unit:63, g_structh.act:698
		for _, st := range glob.Dats.ApDynamicKernel {
			if (st.Kcopy_kernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Fusion_fused_kernel" && len(va) > 1) { // daml.unit:109, g_structh.act:698
		for _, st := range glob.Dats.ApFusion {
			if (st.Kfused_kernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_kernel" && len(va) > 1) { // daml.unit:502, g_structh.act:698
		for _, st := range glob.Dats.ApOp {
			if (st.Kkernelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:46, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Kernel > daml.unit:46, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Kernel > daml.unit:46, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "DynamicKernel" { // daml.unit:58, g_structh.act:676
		for _, st := range me.ItsDynamicKernel {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "KernelParam" { // daml.unit:72, g_structh.act:676
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
	if va[0] == "KernelOp" { // daml.unit:81, g_structh.act:676
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
	if (va[0] == "DynamicKernel_copy_kernel") { // daml.unit:63, g_structh.act:583
		for _, st := range glob.Dats.ApDynamicKernel {
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
	if (va[0] == "Fusion_fused_kernel") { // daml.unit:109, g_structh.act:583
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
	if (va[0] == "Op_kernel") { // daml.unit:502, g_structh.act:583
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
	        fmt.Printf("?No its %s for Kernel %s,%s > daml.unit:46, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDynamicKernel struct {
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
	Kgpu_targetp int
}

func (me KpDynamicKernel) TypeName() string {
    return me.Comp
}
func (me KpDynamicKernel) GetLineNo() string {
	return me.LineNo
}

func loadDynamicKernel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDynamicKernel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDynamicKernel)
	st.LineNo = lno
	st.Comp = "DynamicKernel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcopy_kernelp = -1
	st.Khardwarep = -1
	st.Kgpu_targetp = -1
	st.Kparentp = len( act.ApKernel ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DynamicKernel has no Kernel parent\n") ;
		return 1
	}
	st.Parent = act.ApKernel[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DynamicKernel under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApKernel[ len( act.ApKernel )-1 ].Childs = append(act.ApKernel[ len( act.ApKernel )-1 ].Childs, st)
	act.ApKernel[ len( act.ApKernel )-1 ].ItsDynamicKernel = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsDynamicKernel, st)	// daml.unit:46, g_structh.act:403
	name,_ := st.Names["dynamic_kernel"]
	s := strconv.Itoa(st.Kparentp) + "_DynamicKernel_" + name	// daml.unit:62, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDynamicKernel = append(act.ApDynamicKernel, st)
	return 0
}

func (me KpDynamicKernel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "copy_kernel" { // daml.unit:63, g_structh.act:609
		if (me.Kcopy_kernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kcopy_kernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // daml.unit:64, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "gpu_target" { // daml.unit:65, g_structh.act:609
		if (me.Kgpu_targetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApGPU[ me.Kgpu_targetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:46, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:58, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DynamicKernel > daml.unit:58, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DynamicKernel > daml.unit:58, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDynamicKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:46, g_structh.act:557
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
	if va[0] == "gpu_target" {
		if me.Kgpu_targetp >= 0 {
			st := glob.Dats.ApGPU[ me.Kgpu_targetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DynamicKernel %s,%s > daml.unit:58, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelParam = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelParam, st)	// daml.unit:46, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_KernelParam_" + name	// daml.unit:76, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApKernelParam = append(act.ApKernelParam, st)
	return 0
}

func (me KpKernelParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:46, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:72, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelParam > daml.unit:72, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelParam > daml.unit:72, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:46, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for KernelParam %s,%s > daml.unit:72, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelOp = append(act.ApKernel[ len( act.ApKernel )-1 ].ItsKernelOp, st)	// daml.unit:46, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_KernelOp_" + name	// daml.unit:85, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApKernelOp = append(act.ApKernelOp, st)
	return 0
}

func (me KpKernelOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:46, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:81, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelOp > daml.unit:81, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelOp > daml.unit:81, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:46, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApKernel[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Op_kernel_op") { // daml.unit:503, g_structh.act:583
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
	        fmt.Printf("?No its %s for KernelOp %s,%s > daml.unit:81, g_structh.act:209?", va[0], lno, me.LineNo)
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
	if va[0] == "target" { // daml.unit:98, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:94, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimization[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Optimization > daml.unit:94, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Optimization > daml.unit:94, g_structh.act:187?", va[0], lno, me.LineNo) }
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
	        fmt.Printf("?No its %s for Optimization %s,%s > daml.unit:94, g_structh.act:209?", va[0], lno, me.LineNo)
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
	if va[0] == "fused_kernel" { // daml.unit:109, g_structh.act:609
		if (me.Kfused_kernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kfused_kernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // daml.unit:110, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:103, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusion[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Fusion > daml.unit:103, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Fusion > daml.unit:103, g_structh.act:187?", va[0], lno, me.LineNo) }
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
	        fmt.Printf("?No its %s for Fusion %s,%s > daml.unit:103, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsPyTorch [] *KpPyTorch 
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
	if (va[0] == "Model_framework" && len(va) > 1) { // daml.unit:458, g_structh.act:698
		for _, st := range glob.Dats.ApModel {
			if (st.Kframeworkp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:117, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFramework[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Framework > daml.unit:117, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Framework > daml.unit:117, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFramework) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "PyTorch" { // daml.unit:129, g_structh.act:676
		for _, st := range me.ItsPyTorch {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "Model_framework") { // daml.unit:458, g_structh.act:583
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
	        fmt.Printf("?No its %s for Framework %s,%s > daml.unit:117, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPyTorch struct {
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

func (me KpPyTorch) TypeName() string {
    return me.Comp
}
func (me KpPyTorch) GetLineNo() string {
	return me.LineNo
}

func loadPyTorch(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPyTorch)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPyTorch)
	st.LineNo = lno
	st.Comp = "PyTorch";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApFramework ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PyTorch has no Framework parent\n") ;
		return 1
	}
	st.Parent = act.ApFramework[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PyTorch under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApFramework[ len( act.ApFramework )-1 ].Childs = append(act.ApFramework[ len( act.ApFramework )-1 ].Childs, st)
	act.ApFramework[ len( act.ApFramework )-1 ].ItsPyTorch = append(act.ApFramework[ len( act.ApFramework )-1 ].ItsPyTorch, st)	// daml.unit:117, g_structh.act:403
	name,_ := st.Names["pytorch_id"]
	s := strconv.Itoa(st.Kparentp) + "_PyTorch_" + name	// daml.unit:133, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApPyTorch = append(act.ApPyTorch, st)
	return 0
}

func (me KpPyTorch) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:117, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFramework[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:129, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPyTorch[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PyTorch > daml.unit:129, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PyTorch > daml.unit:129, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPyTorch) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:117, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApFramework[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PyTorch %s,%s > daml.unit:129, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsSearchSpaceCode [] *KpSearchSpaceCode 
	ItsArchitectureParam [] *KpArchitectureParam 
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
	name,_ := st.Names["space"]
	act.index["SearchSpace_" + name] = st.Me;
	st.MyName = name
	act.ApSearchSpace = append(act.ApSearchSpace, st)
	return 0
}

func (me KpSearchSpace) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Model_search_space" && len(va) > 1) { // daml.unit:459, g_structh.act:698
		for _, st := range glob.Dats.ApModel {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ArchitectureSearchOp_search_space" && len(va) > 1) { // daml.unit:522, g_structh.act:698
		for _, st := range glob.Dats.ApArchitectureSearchOp {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchSpaceOp_search_space" && len(va) > 1) { // daml.unit:534, g_structh.act:698
		for _, st := range glob.Dats.ApSearchSpaceOp {
			if (st.Ksearch_spacep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:147, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpace > daml.unit:147, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchSpace > daml.unit:147, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchSpace) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "SearchSpaceCode" { // daml.unit:161, g_structh.act:676
		for _, st := range me.ItsSearchSpaceCode {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "ArchitectureParam" { // daml.unit:170, g_structh.act:676
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
	if (va[0] == "Model_search_space") { // daml.unit:459, g_structh.act:583
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
	if (va[0] == "ArchitectureSearchOp_search_space") { // daml.unit:522, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureSearchOp {
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
	if (va[0] == "SearchSpaceOp_search_space") { // daml.unit:534, g_structh.act:583
		for _, st := range glob.Dats.ApSearchSpaceOp {
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
	        fmt.Printf("?No its %s for SearchSpace %s,%s > daml.unit:147, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchSpaceCode struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Koperations_dictp int
	Kcell_codep int
}

func (me KpSearchSpaceCode) TypeName() string {
    return me.Comp
}
func (me KpSearchSpaceCode) GetLineNo() string {
	return me.LineNo
}

func loadSearchSpaceCode(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchSpaceCode)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchSpaceCode)
	st.LineNo = lno
	st.Comp = "SearchSpaceCode";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Koperations_dictp = -1
	st.Kcell_codep = -1
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchSpaceCode has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchSpaceCode under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchSpaceCode = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchSpaceCode, st)	// daml.unit:147, g_structh.act:403
	name,_ := st.Names["space_code_id"]
	s := strconv.Itoa(st.Kparentp) + "_SearchSpaceCode_" + name	// daml.unit:165, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchSpaceCode = append(act.ApSearchSpaceCode, st)
	return 0
}

func (me KpSearchSpaceCode) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "operations_dict" { // daml.unit:166, g_structh.act:609
		if (me.Koperations_dictp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Koperations_dictp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "cell_code" { // daml.unit:167, g_structh.act:609
		if (me.Kcell_codep >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcell_codep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:147, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:161, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpaceCode[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpaceCode > daml.unit:161, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchSpaceCode > daml.unit:161, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchSpaceCode) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:147, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "operations_dict" {
		if me.Koperations_dictp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Koperations_dictp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "cell_code" {
		if me.Kcell_codep >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcell_codep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchSpaceCode %s,%s > daml.unit:161, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsEnhancedArchitectureParam [] *KpEnhancedArchitectureParam 
	Childs [] Kp
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
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ArchitectureParam has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ArchitectureParam under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsArchitectureParam = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsArchitectureParam, st)	// daml.unit:147, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_ArchitectureParam_" + name	// daml.unit:174, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArchitectureParam = append(act.ApArchitectureParam, st)
	return 0
}

func (me KpArchitectureParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:147, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:170, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArchitectureParam > daml.unit:170, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArchitectureParam > daml.unit:170, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitectureParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "EnhancedArchitectureParam" { // daml.unit:819, g_structh.act:676
		for _, st := range me.ItsEnhancedArchitectureParam {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // daml.unit:147, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "SearchSpaceOp_param_choices") { // daml.unit:535, g_structh.act:583
		for _, st := range glob.Dats.ApSearchSpaceOp {
			if (st.Kparam_choicesp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for ArchitectureParam %s,%s > daml.unit:170, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Khardwarep int
	ItsArchitectureGradient [] *KpArchitectureGradient 
	Childs [] Kp
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
	st.Khardwarep = -1
	name,_ := st.Names["search_op"]
	act.index["SearchOp_" + name] = st.Me;
	st.MyName = name
	act.ApSearchOp = append(act.ApSearchOp, st)
	return 0
}

func (me KpSearchOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // daml.unit:190, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "ArchitectureSearchOp_search_method" && len(va) > 1) { // daml.unit:523, g_structh.act:698
		for _, st := range glob.Dats.ApArchitectureSearchOp {
			if (st.Ksearch_methodp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:182, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchOp > daml.unit:182, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchOp > daml.unit:182, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ArchitectureGradient" { // daml.unit:194, g_structh.act:676
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
	if (va[0] == "ArchitectureSearchOp_search_method") { // daml.unit:523, g_structh.act:583
		for _, st := range glob.Dats.ApArchitectureSearchOp {
			if (st.Ksearch_methodp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for SearchOp %s,%s > daml.unit:182, g_structh.act:209?", va[0], lno, me.LineNo)
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
	st.Kparentp = len( act.ApSearchOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ArchitectureGradient has no SearchOp parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ArchitectureGradient under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchOp[ len( act.ApSearchOp )-1 ].Childs = append(act.ApSearchOp[ len( act.ApSearchOp )-1 ].Childs, st)
	act.ApSearchOp[ len( act.ApSearchOp )-1 ].ItsArchitectureGradient = append(act.ApSearchOp[ len( act.ApSearchOp )-1 ].ItsArchitectureGradient, st)	// daml.unit:182, g_structh.act:403
	name,_ := st.Names["grad_method"]
	s := strconv.Itoa(st.Kparentp) + "_ArchitectureGradient_" + name	// daml.unit:198, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArchitectureGradient = append(act.ApArchitectureGradient, st)
	return 0
}

func (me KpArchitectureGradient) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:182, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:194, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureGradient[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArchitectureGradient > daml.unit:194, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArchitectureGradient > daml.unit:194, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitectureGradient) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:182, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ArchitectureGradient %s,%s > daml.unit:194, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpExpertSystem struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	ItsExpert [] *KpExpert 
	ItsSparseExpertSystem [] *KpSparseExpertSystem 
	Childs [] Kp
}

func (me KpExpertSystem) TypeName() string {
    return me.Comp
}
func (me KpExpertSystem) GetLineNo() string {
	return me.LineNo
}

func loadExpertSystem(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpExpertSystem)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApExpertSystem)
	st.LineNo = lno
	st.Comp = "ExpertSystem";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["system"]
	act.index["ExpertSystem_" + name] = st.Me;
	st.MyName = name
	act.ApExpertSystem = append(act.ApExpertSystem, st)
	return 0
}

func (me KpExpertSystem) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:209, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExpertSystem[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ExpertSystem > daml.unit:209, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ExpertSystem > daml.unit:209, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExpertSystem) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Expert" { // daml.unit:221, g_structh.act:676
		for _, st := range me.ItsExpert {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "SparseExpertSystem" { // daml.unit:264, g_structh.act:676
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
	        fmt.Printf("?No its %s for ExpertSystem %s,%s > daml.unit:209, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpExpert struct {
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

func (me KpExpert) TypeName() string {
    return me.Comp
}
func (me KpExpert) GetLineNo() string {
	return me.LineNo
}

func loadExpert(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpExpert)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApExpert)
	st.LineNo = lno
	st.Comp = "Expert";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApExpertSystem ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Expert has no ExpertSystem parent\n") ;
		return 1
	}
	st.Parent = act.ApExpertSystem[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Expert under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].Childs = append(act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].Childs, st)
	act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].ItsExpert = append(act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].ItsExpert, st)	// daml.unit:209, g_structh.act:403
	name,_ := st.Names["expert"]
	s := strconv.Itoa(st.Kparentp) + "_Expert_" + name	// daml.unit:225, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApExpert = append(act.ApExpert, st)
	return 0
}

func (me KpExpert) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:209, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExpertSystem[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:221, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExpert[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Expert > daml.unit:221, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Expert > daml.unit:221, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExpert) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:209, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApExpertSystem[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Expert %s,%s > daml.unit:221, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpRouterNetwork struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpRouterNetwork) TypeName() string {
    return me.Comp
}
func (me KpRouterNetwork) GetLineNo() string {
	return me.LineNo
}

func loadRouterNetwork(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpRouterNetwork)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApRouterNetwork)
	st.LineNo = lno
	st.Comp = "RouterNetwork";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["router"]
	act.index["RouterNetwork_" + name] = st.Me;
	st.MyName = name
	act.ApRouterNetwork = append(act.ApRouterNetwork, st)
	return 0
}

func (me KpRouterNetwork) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "ExpertRoutingOp_router" && len(va) > 1) { // daml.unit:248, g_structh.act:698
		for _, st := range glob.Dats.ApExpertRoutingOp {
			if (st.Krouterp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:231, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApRouterNetwork[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,RouterNetwork > daml.unit:231, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,RouterNetwork > daml.unit:231, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpRouterNetwork) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "ExpertRoutingOp_router") { // daml.unit:248, g_structh.act:583
		for _, st := range glob.Dats.ApExpertRoutingOp {
			if (st.Krouterp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for RouterNetwork %s,%s > daml.unit:231, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Krouterp int
	Khardwarep int
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
	st.Krouterp = -1
	st.Khardwarep = -1
	name,_ := st.Names["routing_op"]
	act.index["ExpertRoutingOp_" + name] = st.Me;
	st.MyName = name
	act.ApExpertRoutingOp = append(act.ApExpertRoutingOp, st)
	return 0
}

func (me KpExpertRoutingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "router" { // daml.unit:248, g_structh.act:609
		if (me.Krouterp >= 0 && len(va) > 1) {
			return( glob.Dats.ApRouterNetwork[ me.Krouterp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "hardware" { // daml.unit:249, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:243, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExpertRoutingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ExpertRoutingOp > daml.unit:243, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ExpertRoutingOp > daml.unit:243, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExpertRoutingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "CapacityAwareRoutingOp" { // daml.unit:253, g_structh.act:676
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
	if va[0] == "router" {
		if me.Krouterp >= 0 {
			st := glob.Dats.ApRouterNetwork[ me.Krouterp ]
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
	        fmt.Printf("?No its %s for ExpertRoutingOp %s,%s > daml.unit:243, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApExpertRoutingOp[ len( act.ApExpertRoutingOp )-1 ].ItsCapacityAwareRoutingOp = append(act.ApExpertRoutingOp[ len( act.ApExpertRoutingOp )-1 ].ItsCapacityAwareRoutingOp, st)	// daml.unit:243, g_structh.act:403
	name,_ := st.Names["capacity_op"]
	s := strconv.Itoa(st.Kparentp) + "_CapacityAwareRoutingOp_" + name	// daml.unit:257, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCapacityAwareRoutingOp = append(act.ApCapacityAwareRoutingOp, st)
	return 0
}

func (me KpCapacityAwareRoutingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:243, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExpertRoutingOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:253, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCapacityAwareRoutingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CapacityAwareRoutingOp > daml.unit:253, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CapacityAwareRoutingOp > daml.unit:253, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCapacityAwareRoutingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:243, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApExpertRoutingOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CapacityAwareRoutingOp %s,%s > daml.unit:253, g_structh.act:209?", va[0], lno, me.LineNo)
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
	st.Kparentp = len( act.ApExpertSystem ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SparseExpertSystem has no ExpertSystem parent\n") ;
		return 1
	}
	st.Parent = act.ApExpertSystem[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SparseExpertSystem under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].Childs = append(act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].Childs, st)
	act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].ItsSparseExpertSystem = append(act.ApExpertSystem[ len( act.ApExpertSystem )-1 ].ItsSparseExpertSystem, st)	// daml.unit:209, g_structh.act:403
	name,_ := st.Names["sparse_system"]
	s := strconv.Itoa(st.Kparentp) + "_SparseExpertSystem_" + name	// daml.unit:268, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSparseExpertSystem = append(act.ApSparseExpertSystem, st)
	return 0
}

func (me KpSparseExpertSystem) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:209, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExpertSystem[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:264, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSparseExpertSystem[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SparseExpertSystem > daml.unit:264, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SparseExpertSystem > daml.unit:264, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSparseExpertSystem) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:209, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApExpertSystem[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SparseExpertSystem %s,%s > daml.unit:264, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpContinuousModel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpContinuousModel) TypeName() string {
    return me.Comp
}
func (me KpContinuousModel) GetLineNo() string {
	return me.LineNo
}

func loadContinuousModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpContinuousModel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApContinuousModel)
	st.LineNo = lno
	st.Comp = "ContinuousModel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["model"]
	act.index["ContinuousModel_" + name] = st.Me;
	st.MyName = name
	act.ApContinuousModel = append(act.ApContinuousModel, st)
	return 0
}

func (me KpContinuousModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:279, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApContinuousModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ContinuousModel > daml.unit:279, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ContinuousModel > daml.unit:279, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpContinuousModel) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ContinuousModel %s,%s > daml.unit:279, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsNeuralODE [] *KpNeuralODE 
	Childs [] Kp
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
	name,_ := st.Names["ode_op"]
	act.index["ODEOp_" + name] = st.Me;
	st.MyName = name
	act.ApODEOp = append(act.ApODEOp, st)
	return 0
}

func (me KpODEOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:291, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApODEOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ODEOp > daml.unit:291, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ODEOp > daml.unit:291, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpODEOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "NeuralODE" { // daml.unit:302, g_structh.act:676
		for _, st := range me.ItsNeuralODE {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	        fmt.Printf("?No its %s for ODEOp %s,%s > daml.unit:291, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuralODE struct {
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

func (me KpNeuralODE) TypeName() string {
    return me.Comp
}
func (me KpNeuralODE) GetLineNo() string {
	return me.LineNo
}

func loadNeuralODE(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuralODE)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuralODE)
	st.LineNo = lno
	st.Comp = "NeuralODE";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApODEOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralODE has no ODEOp parent\n") ;
		return 1
	}
	st.Parent = act.ApODEOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralODE under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApODEOp[ len( act.ApODEOp )-1 ].Childs = append(act.ApODEOp[ len( act.ApODEOp )-1 ].Childs, st)
	act.ApODEOp[ len( act.ApODEOp )-1 ].ItsNeuralODE = append(act.ApODEOp[ len( act.ApODEOp )-1 ].ItsNeuralODE, st)	// daml.unit:291, g_structh.act:403
	name,_ := st.Names["neural_ode"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralODE_" + name	// daml.unit:306, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralODE = append(act.ApNeuralODE, st)
	return 0
}

func (me KpNeuralODE) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:291, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApODEOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:302, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralODE[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralODE > daml.unit:302, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralODE > daml.unit:302, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralODE) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:291, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApODEOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralODE %s,%s > daml.unit:302, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsNeuralSDE [] *KpNeuralSDE 
	Childs [] Kp
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
	name,_ := st.Names["sde_op"]
	act.index["SDEOp_" + name] = st.Me;
	st.MyName = name
	act.ApSDEOp = append(act.ApSDEOp, st)
	return 0
}

func (me KpSDEOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:313, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSDEOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SDEOp > daml.unit:313, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SDEOp > daml.unit:313, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSDEOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "NeuralSDE" { // daml.unit:324, g_structh.act:676
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
	        fmt.Printf("?No its %s for SDEOp %s,%s > daml.unit:313, g_structh.act:209?", va[0], lno, me.LineNo)
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
	st.Kparentp = len( act.ApSDEOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralSDE has no SDEOp parent\n") ;
		return 1
	}
	st.Parent = act.ApSDEOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralSDE under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSDEOp[ len( act.ApSDEOp )-1 ].Childs = append(act.ApSDEOp[ len( act.ApSDEOp )-1 ].Childs, st)
	act.ApSDEOp[ len( act.ApSDEOp )-1 ].ItsNeuralSDE = append(act.ApSDEOp[ len( act.ApSDEOp )-1 ].ItsNeuralSDE, st)	// daml.unit:313, g_structh.act:403
	name,_ := st.Names["neural_sde"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralSDE_" + name	// daml.unit:328, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralSDE = append(act.ApNeuralSDE, st)
	return 0
}

func (me KpNeuralSDE) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:313, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSDEOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:324, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralSDE[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralSDE > daml.unit:324, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralSDE > daml.unit:324, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralSDE) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:313, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSDEOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralSDE %s,%s > daml.unit:324, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsNeuralPDE [] *KpNeuralPDE 
	Childs [] Kp
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
	name,_ := st.Names["pde_op"]
	act.index["PDEOp_" + name] = st.Me;
	st.MyName = name
	act.ApPDEOp = append(act.ApPDEOp, st)
	return 0
}

func (me KpPDEOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:335, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPDEOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PDEOp > daml.unit:335, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PDEOp > daml.unit:335, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPDEOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "NeuralPDE" { // daml.unit:346, g_structh.act:676
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
	        fmt.Printf("?No its %s for PDEOp %s,%s > daml.unit:335, g_structh.act:209?", va[0], lno, me.LineNo)
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
	st.Kparentp = len( act.ApPDEOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuralPDE has no PDEOp parent\n") ;
		return 1
	}
	st.Parent = act.ApPDEOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuralPDE under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApPDEOp[ len( act.ApPDEOp )-1 ].Childs = append(act.ApPDEOp[ len( act.ApPDEOp )-1 ].Childs, st)
	act.ApPDEOp[ len( act.ApPDEOp )-1 ].ItsNeuralPDE = append(act.ApPDEOp[ len( act.ApPDEOp )-1 ].ItsNeuralPDE, st)	// daml.unit:335, g_structh.act:403
	name,_ := st.Names["neural_pde"]
	s := strconv.Itoa(st.Kparentp) + "_NeuralPDE_" + name	// daml.unit:350, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApNeuralPDE = append(act.ApNeuralPDE, st)
	return 0
}

func (me KpNeuralPDE) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:335, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApPDEOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:346, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuralPDE[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuralPDE > daml.unit:346, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuralPDE > daml.unit:346, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuralPDE) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:335, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApPDEOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuralPDE %s,%s > daml.unit:346, g_structh.act:209?", va[0], lno, me.LineNo)
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
	name,_ := st.Names["depth_op"]
	act.index["ContinuousDepthOp_" + name] = st.Me;
	st.MyName = name
	act.ApContinuousDepthOp = append(act.ApContinuousDepthOp, st)
	return 0
}

func (me KpContinuousDepthOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:357, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApContinuousDepthOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ContinuousDepthOp > daml.unit:357, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ContinuousDepthOp > daml.unit:357, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpContinuousDepthOp) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ContinuousDepthOp %s,%s > daml.unit:357, g_structh.act:209?", va[0], lno, me.LineNo)
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
	name,_ := st.Names["meta_learner"]
	act.index["MetaLearner_" + name] = st.Me;
	st.MyName = name
	act.ApMetaLearner = append(act.ApMetaLearner, st)
	return 0
}

func (me KpMetaLearner) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:371, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMetaLearner[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MetaLearner > daml.unit:371, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MetaLearner > daml.unit:371, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMetaLearner) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for MetaLearner %s,%s > daml.unit:371, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHypernetwork struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpHypernetwork) TypeName() string {
    return me.Comp
}
func (me KpHypernetwork) GetLineNo() string {
	return me.LineNo
}

func loadHypernetwork(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpHypernetwork)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHypernetwork)
	st.LineNo = lno
	st.Comp = "Hypernetwork";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["hypernetwork"]
	act.index["Hypernetwork_" + name] = st.Me;
	st.MyName = name
	act.ApHypernetwork = append(act.ApHypernetwork, st)
	return 0
}

func (me KpHypernetwork) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "WeightGenerationOp_hypernetwork" && len(va) > 1) { // daml.unit:398, g_structh.act:698
		for _, st := range glob.Dats.ApWeightGenerationOp {
			if (st.Khypernetworkp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:382, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHypernetwork[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Hypernetwork > daml.unit:382, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Hypernetwork > daml.unit:382, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHypernetwork) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "WeightGenerationOp_hypernetwork") { // daml.unit:398, g_structh.act:583
		for _, st := range glob.Dats.ApWeightGenerationOp {
			if (st.Khypernetworkp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for Hypernetwork %s,%s > daml.unit:382, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Khypernetworkp int
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
	st.Khypernetworkp = -1
	name,_ := st.Names["weight_gen_op"]
	act.index["WeightGenerationOp_" + name] = st.Me;
	st.MyName = name
	act.ApWeightGenerationOp = append(act.ApWeightGenerationOp, st)
	return 0
}

func (me KpWeightGenerationOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hypernetwork" { // daml.unit:398, g_structh.act:609
		if (me.Khypernetworkp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHypernetwork[ me.Khypernetworkp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:393, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApWeightGenerationOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,WeightGenerationOp > daml.unit:393, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,WeightGenerationOp > daml.unit:393, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpWeightGenerationOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "hypernetwork" {
		if me.Khypernetworkp >= 0 {
			st := glob.Dats.ApHypernetwork[ me.Khypernetworkp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for WeightGenerationOp %s,%s > daml.unit:393, g_structh.act:209?", va[0], lno, me.LineNo)
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
	name,_ := st.Names["program"]
	act.index["DifferentiableProgram_" + name] = st.Me;
	st.MyName = name
	act.ApDifferentiableProgram = append(act.ApDifferentiableProgram, st)
	return 0
}

func (me KpDifferentiableProgram) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:403, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDifferentiableProgram[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DifferentiableProgram > daml.unit:403, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DifferentiableProgram > daml.unit:403, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDifferentiableProgram) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DifferentiableProgram %s,%s > daml.unit:403, g_structh.act:209?", va[0], lno, me.LineNo)
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
	name,_ := st.Names["control"]
	act.index["ControlFlow_" + name] = st.Me;
	st.MyName = name
	act.ApControlFlow = append(act.ApControlFlow, st)
	return 0
}

func (me KpControlFlow) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:418, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ControlFlow > daml.unit:418, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ControlFlow > daml.unit:418, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpControlFlow) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Condition" { // daml.unit:428, g_structh.act:676
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
	if va[0] == "Branch" { // daml.unit:437, g_structh.act:676
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
	        fmt.Printf("?No its %s for ControlFlow %s,%s > daml.unit:418, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsCondition = append(act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsCondition, st)	// daml.unit:418, g_structh.act:403
	name,_ := st.Names["condition"]
	s := strconv.Itoa(st.Kparentp) + "_Condition_" + name	// daml.unit:432, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCondition = append(act.ApCondition, st)
	return 0
}

func (me KpCondition) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:418, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "Branch_condition" && len(va) > 1) { // daml.unit:442, g_structh.act:698
		for _, st := range glob.Dats.ApBranch {
			if (st.Kconditionp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:428, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCondition[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Condition > daml.unit:428, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Condition > daml.unit:428, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCondition) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:418, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Branch_condition") { // daml.unit:442, g_structh.act:583
		for _, st := range glob.Dats.ApBranch {
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
	        fmt.Printf("?No its %s for Condition %s,%s > daml.unit:428, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kconditionp int
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
	st.Kconditionp = -1
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
	act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsBranch = append(act.ApControlFlow[ len( act.ApControlFlow )-1 ].ItsBranch, st)	// daml.unit:418, g_structh.act:403
	name,_ := st.Names["branch"]
	s := strconv.Itoa(st.Kparentp) + "_Branch_" + name	// daml.unit:441, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBranch = append(act.ApBranch, st)
	return 0
}

func (me KpBranch) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "condition" { // daml.unit:442, g_structh.act:609
		if (me.Kconditionp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCondition[ me.Kconditionp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:418, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApControlFlow[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:437, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBranch[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Branch > daml.unit:437, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Branch > daml.unit:437, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBranch) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:418, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApControlFlow[ me.Kparentp ]
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
	        fmt.Printf("?No its %s for Branch %s,%s > daml.unit:437, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Ksearch_spacep int
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
	st.Ksearch_spacep = -1
	name,_ := st.Names["model"]
	act.index["Model_" + name] = st.Me;
	st.MyName = name
	act.ApModel = append(act.ApModel, st)
	return 0
}

func (me KpModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // daml.unit:457, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "framework" { // daml.unit:458, g_structh.act:609
		if (me.Kframeworkp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFramework[ me.Kframeworkp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "search_space" { // daml.unit:459, g_structh.act:609
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Validation_target" && len(va) > 1) { // daml.unit:573, g_structh.act:698
		for _, st := range glob.Dats.ApValidation {
			if (st.Ktargetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Project_model" && len(va) > 1) { // daml.unit:629, g_structh.act:698
		for _, st := range glob.Dats.ApProject {
			if (st.Kmodelp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:451, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Model > daml.unit:451, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Model > daml.unit:451, g_structh.act:187?", va[0], lno, me.LineNo) }
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
	if (va[0] == "Validation_target") { // daml.unit:573, g_structh.act:583
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
	if (va[0] == "Project_model") { // daml.unit:629, g_structh.act:583
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
	        fmt.Printf("?No its %s for Model %s,%s > daml.unit:451, g_structh.act:209?", va[0], lno, me.LineNo)
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
	name,_ := st.Names["tensor"]
	act.index["Tensor_" + name] = st.Me;
	st.MyName = name
	act.ApTensor = append(act.ApTensor, st)
	return 0
}

func (me KpTensor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Arg_tensor" && len(va) > 1) { // daml.unit:512, g_structh.act:698
		for _, st := range glob.Dats.ApArg {
			if (st.Ktensorp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:463, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Tensor > daml.unit:463, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Tensor > daml.unit:463, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTensor) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Arg_tensor") { // daml.unit:512, g_structh.act:583
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
	        fmt.Printf("?No its %s for Tensor %s,%s > daml.unit:463, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kcode_blockp int
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
	st.Kcode_blockp = -1
	name,_ := st.Names["layer"]
	act.index["Layer_" + name] = st.Me;
	st.MyName = name
	act.ApLayer = append(act.ApLayer, st)
	return 0
}

func (me KpLayer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:487, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Schedule_layer" && len(va) > 1) { // daml.unit:559, g_structh.act:698
		for _, st := range glob.Dats.ApSchedule {
			if (st.Klayerp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:475, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Layer > daml.unit:475, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Layer > daml.unit:475, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpLayer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Op" { // daml.unit:494, g_structh.act:676
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
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Schedule_layer") { // daml.unit:559, g_structh.act:583
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
	        fmt.Printf("?No its %s for Layer %s,%s > daml.unit:475, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kcode_blockp int
	ItsArg [] *KpArg 
	ItsArchitectureSearchOp [] *KpArchitectureSearchOp 
	ItsSearchSpaceOp [] *KpSearchSpaceOp 
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
	st.Kcode_blockp = -1
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
	act.ApLayer[ len( act.ApLayer )-1 ].ItsOp = append(act.ApLayer[ len( act.ApLayer )-1 ].ItsOp, st)	// daml.unit:475, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_Op_" + name	// daml.unit:498, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOp = append(act.ApOp, st)
	return 0
}

func (me KpOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "op_type" { // daml.unit:499, g_structh.act:619
		if (me.Kop_typep >= 0 && len(va) > 1) {
			return( me.Childs[ me.Kop_typep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel" { // daml.unit:502, g_structh.act:609
		if (me.Kkernelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kkernelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel_op" { // daml.unit:503, g_structh.act:609
		if (me.Kkernel_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernelOp[ me.Kkernel_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "code_block" { // daml.unit:504, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:475, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:494, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Op > daml.unit:494, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Op > daml.unit:494, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Arg" { // daml.unit:506, g_structh.act:676
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
	if va[0] == "ArchitectureSearchOp" { // daml.unit:517, g_structh.act:676
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
	if va[0] == "SearchSpaceOp" { // daml.unit:528, g_structh.act:676
		for _, st := range me.ItsSearchSpaceOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // daml.unit:475, g_structh.act:557
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
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Schedule_op") { // daml.unit:560, g_structh.act:583
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
	        fmt.Printf("?No its %s for Op %s,%s > daml.unit:494, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApOp[ len( act.ApOp )-1 ].ItsArg = append(act.ApOp[ len( act.ApOp )-1 ].ItsArg, st)	// daml.unit:494, g_structh.act:403
	name,_ := st.Names["arg"]
	s := strconv.Itoa(st.Kparentp) + "_Arg_" + name	// daml.unit:510, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArg = append(act.ApArg, st)
	return 0
}

func (me KpArg) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "tensor" { // daml.unit:512, g_structh.act:609
		if (me.Ktensorp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTensor[ me.Ktensorp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:494, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:506, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArg[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Arg > daml.unit:506, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Arg > daml.unit:506, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArg) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:494, g_structh.act:557
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
	        fmt.Printf("?No its %s for Arg %s,%s > daml.unit:506, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Ksearch_spacep int
	Ksearch_methodp int
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
	st.Ksearch_spacep = -1
	st.Ksearch_methodp = -1
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
	act.ApOp[ len( act.ApOp )-1 ].ItsArchitectureSearchOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsArchitectureSearchOp, st)	// daml.unit:494, g_structh.act:403
	name,_ := st.Names["search_op"]
	s := strconv.Itoa(st.Kparentp) + "_ArchitectureSearchOp_" + name	// daml.unit:521, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApArchitectureSearchOp = append(act.ApArchitectureSearchOp, st)
	return 0
}

func (me KpArchitectureSearchOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "search_space" { // daml.unit:522, g_structh.act:609
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "search_method" { // daml.unit:523, g_structh.act:609
		if (me.Ksearch_methodp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchOp[ me.Ksearch_methodp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:494, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:517, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureSearchOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ArchitectureSearchOp > daml.unit:517, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ArchitectureSearchOp > daml.unit:517, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitectureSearchOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:494, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
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
	if va[0] == "search_method" {
		if me.Ksearch_methodp >= 0 {
			st := glob.Dats.ApSearchOp[ me.Ksearch_methodp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ArchitectureSearchOp %s,%s > daml.unit:517, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchSpaceOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Ksearch_spacep int
	Kparam_choicesp int
}

func (me KpSearchSpaceOp) TypeName() string {
    return me.Comp
}
func (me KpSearchSpaceOp) GetLineNo() string {
	return me.LineNo
}

func loadSearchSpaceOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchSpaceOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchSpaceOp)
	st.LineNo = lno
	st.Comp = "SearchSpaceOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ksearch_spacep = -1
	st.Kparam_choicesp = -1
	st.Kparentp = len( act.ApOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchSpaceOp has no Op parent\n") ;
		return 1
	}
	st.Parent = act.ApOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchSpaceOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOp[ len( act.ApOp )-1 ].Childs = append(act.ApOp[ len( act.ApOp )-1 ].Childs, st)
	act.ApOp[ len( act.ApOp )-1 ].ItsSearchSpaceOp = append(act.ApOp[ len( act.ApOp )-1 ].ItsSearchSpaceOp, st)	// daml.unit:494, g_structh.act:403
	name,_ := st.Names["space_op"]
	s := strconv.Itoa(st.Kparentp) + "_SearchSpaceOp_" + name	// daml.unit:532, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchSpaceOp = append(act.ApSearchSpaceOp, st)
	return 0
}

func (me KpSearchSpaceOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "search_space" { // daml.unit:534, g_structh.act:609
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "param_choices" { // daml.unit:535, g_structh.act:609
		if (me.Kparam_choicesp >= 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureParam[ me.Kparam_choicesp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:494, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:528, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpaceOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpaceOp > daml.unit:528, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchSpaceOp > daml.unit:528, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchSpaceOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:494, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOp[ me.Kparentp ]
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
	if va[0] == "param_choices" {
		if me.Kparam_choicesp >= 0 {
			st := glob.Dats.ApArchitectureParam[ me.Kparam_choicesp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchSpaceOp %s,%s > daml.unit:528, g_structh.act:209?", va[0], lno, me.LineNo)
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
	if va[0] == "target" { // daml.unit:548, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "TargetConfig_config" && len(va) > 1) { // daml.unit:639, g_structh.act:698
		for _, st := range glob.Dats.ApTargetConfig {
			if (st.Kconfigp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:543, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Config > daml.unit:543, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Config > daml.unit:543, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Schedule" { // daml.unit:554, g_structh.act:676
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
	if (va[0] == "TargetConfig_config") { // daml.unit:639, g_structh.act:583
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
	        fmt.Printf("?No its %s for Config %s,%s > daml.unit:543, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApConfig[ len( act.ApConfig )-1 ].ItsSchedule = append(act.ApConfig[ len( act.ApConfig )-1 ].ItsSchedule, st)	// daml.unit:543, g_structh.act:403
	act.ApSchedule = append(act.ApSchedule, st)
	return 0
}

func (me KpSchedule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "layer" { // daml.unit:559, g_structh.act:609
		if (me.Klayerp >= 0 && len(va) > 1) {
			return( glob.Dats.ApLayer[ me.Klayerp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "op" { // daml.unit:560, g_structh.act:609
		if (me.Kopp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOp[ me.Kopp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:543, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:554, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSchedule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Schedule > daml.unit:554, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Schedule > daml.unit:554, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSchedule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:543, g_structh.act:557
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
	        fmt.Printf("?No its %s for Schedule %s,%s > daml.unit:554, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsDynamicConstraint [] *KpDynamicConstraint 
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
	if va[0] == "target" { // daml.unit:573, g_structh.act:609
		if (me.Ktargetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Ktargetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:568, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApValidation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Validation > daml.unit:568, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Validation > daml.unit:568, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpValidation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "DynamicConstraint" { // daml.unit:577, g_structh.act:676
		for _, st := range me.ItsDynamicConstraint {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	        fmt.Printf("?No its %s for Validation %s,%s > daml.unit:568, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDynamicConstraint struct {
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

func (me KpDynamicConstraint) TypeName() string {
    return me.Comp
}
func (me KpDynamicConstraint) GetLineNo() string {
	return me.LineNo
}

func loadDynamicConstraint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDynamicConstraint)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDynamicConstraint)
	st.LineNo = lno
	st.Comp = "DynamicConstraint";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApValidation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DynamicConstraint has no Validation parent\n") ;
		return 1
	}
	st.Parent = act.ApValidation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DynamicConstraint under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApValidation[ len( act.ApValidation )-1 ].Childs = append(act.ApValidation[ len( act.ApValidation )-1 ].Childs, st)
	act.ApValidation[ len( act.ApValidation )-1 ].ItsDynamicConstraint = append(act.ApValidation[ len( act.ApValidation )-1 ].ItsDynamicConstraint, st)	// daml.unit:568, g_structh.act:403
	name,_ := st.Names["constraint"]
	s := strconv.Itoa(st.Kparentp) + "_DynamicConstraint_" + name	// daml.unit:581, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDynamicConstraint = append(act.ApDynamicConstraint, st)
	return 0
}

func (me KpDynamicConstraint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:568, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApValidation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:577, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDynamicConstraint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DynamicConstraint > daml.unit:577, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DynamicConstraint > daml.unit:577, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDynamicConstraint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:568, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApValidation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DynamicConstraint %s,%s > daml.unit:577, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchMethodRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpSearchMethodRule) TypeName() string {
    return me.Comp
}
func (me KpSearchMethodRule) GetLineNo() string {
	return me.LineNo
}

func loadSearchMethodRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchMethodRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchMethodRule)
	st.LineNo = lno
	st.Comp = "SearchMethodRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["method"]
	act.index["SearchMethodRule_" + name] = st.Me;
	st.MyName = name
	act.ApSearchMethodRule = append(act.ApSearchMethodRule, st)
	return 0
}

func (me KpSearchMethodRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:592, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchMethodRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchMethodRule > daml.unit:592, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchMethodRule > daml.unit:592, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchMethodRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for SearchMethodRule %s,%s > daml.unit:592, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpRoutingStrategyRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpRoutingStrategyRule) TypeName() string {
    return me.Comp
}
func (me KpRoutingStrategyRule) GetLineNo() string {
	return me.LineNo
}

func loadRoutingStrategyRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpRoutingStrategyRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApRoutingStrategyRule)
	st.LineNo = lno
	st.Comp = "RoutingStrategyRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["strategy"]
	act.index["RoutingStrategyRule_" + name] = st.Me;
	st.MyName = name
	act.ApRoutingStrategyRule = append(act.ApRoutingStrategyRule, st)
	return 0
}

func (me KpRoutingStrategyRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:601, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApRoutingStrategyRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,RoutingStrategyRule > daml.unit:601, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,RoutingStrategyRule > daml.unit:601, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpRoutingStrategyRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for RoutingStrategyRule %s,%s > daml.unit:601, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSolverRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpSolverRule) TypeName() string {
    return me.Comp
}
func (me KpSolverRule) GetLineNo() string {
	return me.LineNo
}

func loadSolverRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSolverRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSolverRule)
	st.LineNo = lno
	st.Comp = "SolverRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["solver"]
	act.index["SolverRule_" + name] = st.Me;
	st.MyName = name
	act.ApSolverRule = append(act.ApSolverRule, st)
	return 0
}

func (me KpSolverRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:610, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSolverRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SolverRule > daml.unit:610, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SolverRule > daml.unit:610, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSolverRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for SolverRule %s,%s > daml.unit:610, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsCodegenRule [] *KpCodegenRule 
	ItsProjectValidation [] *KpProjectValidation 
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
	if va[0] == "domain" { // daml.unit:628, g_structh.act:609
		if (me.Kdomainp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDomain[ me.Kdomainp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "model" { // daml.unit:629, g_structh.act:609
		if (me.Kmodelp >= 0 && len(va) > 1) {
			return( glob.Dats.ApModel[ me.Kmodelp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "BuildRule_project" && len(va) > 1) { // daml.unit:652, g_structh.act:698
		for _, st := range glob.Dats.ApBuildRule {
			if (st.Kprojectp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:623, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Project > daml.unit:623, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Project > daml.unit:623, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProject) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TargetConfig" { // daml.unit:632, g_structh.act:676
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
	if va[0] == "BuildRule" { // daml.unit:644, g_structh.act:676
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
	if va[0] == "CodegenRule" { // daml.unit:657, g_structh.act:676
		for _, st := range me.ItsCodegenRule {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "ProjectValidation" { // daml.unit:850, g_structh.act:676
		for _, st := range me.ItsProjectValidation {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "BuildRule_project") { // daml.unit:652, g_structh.act:583
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
	        fmt.Printf("?No its %s for Project %s,%s > daml.unit:623, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApProject[ len( act.ApProject )-1 ].ItsTargetConfig = append(act.ApProject[ len( act.ApProject )-1 ].ItsTargetConfig, st)	// daml.unit:623, g_structh.act:403
	name,_ := st.Names["target_id"]
	s := strconv.Itoa(st.Kparentp) + "_TargetConfig_" + name	// daml.unit:636, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTargetConfig = append(act.ApTargetConfig, st)
	return 0
}

func (me KpTargetConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // daml.unit:637, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "config" { // daml.unit:639, g_structh.act:609
		if (me.Kconfigp >= 0 && len(va) > 1) {
			return( glob.Dats.ApConfig[ me.Kconfigp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:623, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:632, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTargetConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TargetConfig > daml.unit:632, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TargetConfig > daml.unit:632, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTargetConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:623, g_structh.act:557
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
	if (va[0] == "BuildRule_validate_against") { // daml.unit:653, g_structh.act:583
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
	        fmt.Printf("?No its %s for TargetConfig %s,%s > daml.unit:632, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApProject[ len( act.ApProject )-1 ].ItsBuildRule = append(act.ApProject[ len( act.ApProject )-1 ].ItsBuildRule, st)	// daml.unit:623, g_structh.act:403
	name,_ := st.Names["build_id"]
	s := strconv.Itoa(st.Kparentp) + "_BuildRule_" + name	// daml.unit:648, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApBuildRule = append(act.ApBuildRule, st)
	return 0
}

func (me KpBuildRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project" { // daml.unit:652, g_structh.act:609
		if (me.Kprojectp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kprojectp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "validate_against" { // daml.unit:653, g_structh.act:609
		if (me.Kvalidate_againstp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTargetConfig[ me.Kvalidate_againstp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:623, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:644, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApBuildRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,BuildRule > daml.unit:644, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,BuildRule > daml.unit:644, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpBuildRule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:623, g_structh.act:557
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
	        fmt.Printf("?No its %s for BuildRule %s,%s > daml.unit:644, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCodegenRule struct {
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

func (me KpCodegenRule) TypeName() string {
    return me.Comp
}
func (me KpCodegenRule) GetLineNo() string {
	return me.LineNo
}

func loadCodegenRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCodegenRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCodegenRule)
	st.LineNo = lno
	st.Comp = "CodegenRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " CodegenRule has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " CodegenRule under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsCodegenRule = append(act.ApProject[ len( act.ApProject )-1 ].ItsCodegenRule, st)	// daml.unit:623, g_structh.act:403
	name,_ := st.Names["gen_id"]
	s := strconv.Itoa(st.Kparentp) + "_CodegenRule_" + name	// daml.unit:661, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApCodegenRule = append(act.ApCodegenRule, st)
	return 0
}

func (me KpCodegenRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:623, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:657, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCodegenRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CodegenRule > daml.unit:657, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CodegenRule > daml.unit:657, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCodegenRule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:623, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CodegenRule %s,%s > daml.unit:657, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpStructuredChoice struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpStructuredChoice) TypeName() string {
    return me.Comp
}
func (me KpStructuredChoice) GetLineNo() string {
	return me.LineNo
}

func loadStructuredChoice(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpStructuredChoice)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApStructuredChoice)
	st.LineNo = lno
	st.Comp = "StructuredChoice";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["choice_id"]
	act.index["StructuredChoice_" + name] = st.Me;
	st.MyName = name
	act.ApStructuredChoice = append(act.ApStructuredChoice, st)
	return 0
}

func (me KpStructuredChoice) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "EnhancedArchitectureParam_structured_choices" && len(va) > 1) { // daml.unit:823, g_structh.act:698
		for _, st := range glob.Dats.ApEnhancedArchitectureParam {
			if (st.Kstructured_choicesp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:677, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApStructuredChoice[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,StructuredChoice > daml.unit:677, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,StructuredChoice > daml.unit:677, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpStructuredChoice) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "EnhancedArchitectureParam_structured_choices") { // daml.unit:823, g_structh.act:583
		for _, st := range glob.Dats.ApEnhancedArchitectureParam {
			if (st.Kstructured_choicesp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for StructuredChoice %s,%s > daml.unit:677, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNumericRange struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpNumericRange) TypeName() string {
    return me.Comp
}
func (me KpNumericRange) GetLineNo() string {
	return me.LineNo
}

func loadNumericRange(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNumericRange)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNumericRange)
	st.LineNo = lno
	st.Comp = "NumericRange";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["range_id"]
	act.index["NumericRange_" + name] = st.Me;
	st.MyName = name
	act.ApNumericRange = append(act.ApNumericRange, st)
	return 0
}

func (me KpNumericRange) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "EnhancedArchitectureParam_numeric_range" && len(va) > 1) { // daml.unit:824, g_structh.act:698
		for _, st := range glob.Dats.ApEnhancedArchitectureParam {
			if (st.Knumeric_rangep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:687, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNumericRange[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NumericRange > daml.unit:687, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NumericRange > daml.unit:687, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNumericRange) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "EnhancedArchitectureParam_numeric_range") { // daml.unit:824, g_structh.act:583
		for _, st := range glob.Dats.ApEnhancedArchitectureParam {
			if (st.Knumeric_rangep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for NumericRange %s,%s > daml.unit:687, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpParameterMap struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpParameterMap) TypeName() string {
    return me.Comp
}
func (me KpParameterMap) GetLineNo() string {
	return me.LineNo
}

func loadParameterMap(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpParameterMap)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApParameterMap)
	st.LineNo = lno
	st.Comp = "ParameterMap";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["map_id"]
	act.index["ParameterMap_" + name] = st.Me;
	st.MyName = name
	act.ApParameterMap = append(act.ApParameterMap, st)
	return 0
}

func (me KpParameterMap) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:699, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApParameterMap[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ParameterMap > daml.unit:699, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ParameterMap > daml.unit:699, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpParameterMap) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ParameterMap %s,%s > daml.unit:699, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCostModel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpCostModel) TypeName() string {
    return me.Comp
}
func (me KpCostModel) GetLineNo() string {
	return me.LineNo
}

func loadCostModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCostModel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCostModel)
	st.LineNo = lno
	st.Comp = "CostModel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["cost_id"]
	act.index["CostModel_" + name] = st.Me;
	st.MyName = name
	act.ApCostModel = append(act.ApCostModel, st)
	return 0
}

func (me KpCostModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:709, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCostModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CostModel > daml.unit:709, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CostModel > daml.unit:709, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCostModel) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for CostModel %s,%s > daml.unit:709, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpExecutionContext struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpExecutionContext) TypeName() string {
    return me.Comp
}
func (me KpExecutionContext) GetLineNo() string {
	return me.LineNo
}

func loadExecutionContext(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpExecutionContext)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApExecutionContext)
	st.LineNo = lno
	st.Comp = "ExecutionContext";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["context_id"]
	act.index["ExecutionContext_" + name] = st.Me;
	st.MyName = name
	act.ApExecutionContext = append(act.ApExecutionContext, st)
	return 0
}

func (me KpExecutionContext) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:724, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExecutionContext[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ExecutionContext > daml.unit:724, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ExecutionContext > daml.unit:724, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExecutionContext) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ExecutionContext %s,%s > daml.unit:724, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMemoryBudget struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpMemoryBudget) TypeName() string {
    return me.Comp
}
func (me KpMemoryBudget) GetLineNo() string {
	return me.LineNo
}

func loadMemoryBudget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMemoryBudget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMemoryBudget)
	st.LineNo = lno
	st.Comp = "MemoryBudget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["budget_id"]
	act.index["MemoryBudget_" + name] = st.Me;
	st.MyName = name
	act.ApMemoryBudget = append(act.ApMemoryBudget, st)
	return 0
}

func (me KpMemoryBudget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:735, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMemoryBudget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MemoryBudget > daml.unit:735, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MemoryBudget > daml.unit:735, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMemoryBudget) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for MemoryBudget %s,%s > daml.unit:735, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDataType struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDataType) TypeName() string {
    return me.Comp
}
func (me KpDataType) GetLineNo() string {
	return me.LineNo
}

func loadDataType(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDataType)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDataType)
	st.LineNo = lno
	st.Comp = "DataType";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["dtype"]
	act.index["DataType_" + name] = st.Me;
	st.MyName = name
	act.ApDataType = append(act.ApDataType, st)
	return 0
}

func (me KpDataType) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:750, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDataType[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DataType > daml.unit:750, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DataType > daml.unit:750, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDataType) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DataType %s,%s > daml.unit:750, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpShapeConstraint struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpShapeConstraint) TypeName() string {
    return me.Comp
}
func (me KpShapeConstraint) GetLineNo() string {
	return me.LineNo
}

func loadShapeConstraint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpShapeConstraint)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApShapeConstraint)
	st.LineNo = lno
	st.Comp = "ShapeConstraint";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["constraint_id"]
	act.index["ShapeConstraint_" + name] = st.Me;
	st.MyName = name
	act.ApShapeConstraint = append(act.ApShapeConstraint, st)
	return 0
}

func (me KpShapeConstraint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:760, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApShapeConstraint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ShapeConstraint > daml.unit:760, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ShapeConstraint > daml.unit:760, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpShapeConstraint) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ShapeConstraint %s,%s > daml.unit:760, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCompatibilityRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpCompatibilityRule) TypeName() string {
    return me.Comp
}
func (me KpCompatibilityRule) GetLineNo() string {
	return me.LineNo
}

func loadCompatibilityRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCompatibilityRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCompatibilityRule)
	st.LineNo = lno
	st.Comp = "CompatibilityRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["rule_id"]
	act.index["CompatibilityRule_" + name] = st.Me;
	st.MyName = name
	act.ApCompatibilityRule = append(act.ApCompatibilityRule, st)
	return 0
}

func (me KpCompatibilityRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:771, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCompatibilityRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CompatibilityRule > daml.unit:771, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CompatibilityRule > daml.unit:771, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCompatibilityRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for CompatibilityRule %s,%s > daml.unit:771, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpValidatedReference struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpValidatedReference) TypeName() string {
    return me.Comp
}
func (me KpValidatedReference) GetLineNo() string {
	return me.LineNo
}

func loadValidatedReference(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpValidatedReference)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApValidatedReference)
	st.LineNo = lno
	st.Comp = "ValidatedReference";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["ref_id"]
	act.index["ValidatedReference_" + name] = st.Me;
	st.MyName = name
	act.ApValidatedReference = append(act.ApValidatedReference, st)
	return 0
}

func (me KpValidatedReference) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:787, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApValidatedReference[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ValidatedReference > daml.unit:787, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ValidatedReference > daml.unit:787, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpValidatedReference) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ValidatedReference %s,%s > daml.unit:787, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMathExpression struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpMathExpression) TypeName() string {
    return me.Comp
}
func (me KpMathExpression) GetLineNo() string {
	return me.LineNo
}

func loadMathExpression(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMathExpression)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMathExpression)
	st.LineNo = lno
	st.Comp = "MathExpression";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["expression_id"]
	act.index["MathExpression_" + name] = st.Me;
	st.MyName = name
	act.ApMathExpression = append(act.ApMathExpression, st)
	return 0
}

func (me KpMathExpression) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:802, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMathExpression[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MathExpression > daml.unit:802, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MathExpression > daml.unit:802, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMathExpression) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for MathExpression %s,%s > daml.unit:802, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnhancedArchitectureParam struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kstructured_choicesp int
	Knumeric_rangep int
}

func (me KpEnhancedArchitectureParam) TypeName() string {
    return me.Comp
}
func (me KpEnhancedArchitectureParam) GetLineNo() string {
	return me.LineNo
}

func loadEnhancedArchitectureParam(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEnhancedArchitectureParam)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnhancedArchitectureParam)
	st.LineNo = lno
	st.Comp = "EnhancedArchitectureParam";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kstructured_choicesp = -1
	st.Knumeric_rangep = -1
	st.Kparentp = len( act.ApArchitectureParam ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " EnhancedArchitectureParam has no ArchitectureParam parent\n") ;
		return 1
	}
	st.Parent = act.ApArchitectureParam[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " EnhancedArchitectureParam under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApArchitectureParam[ len( act.ApArchitectureParam )-1 ].Childs = append(act.ApArchitectureParam[ len( act.ApArchitectureParam )-1 ].Childs, st)
	act.ApArchitectureParam[ len( act.ApArchitectureParam )-1 ].ItsEnhancedArchitectureParam = append(act.ApArchitectureParam[ len( act.ApArchitectureParam )-1 ].ItsEnhancedArchitectureParam, st)	// daml.unit:170, g_structh.act:403
	act.ApEnhancedArchitectureParam = append(act.ApEnhancedArchitectureParam, st)
	return 0
}

func (me KpEnhancedArchitectureParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "structured_choices" { // daml.unit:823, g_structh.act:609
		if (me.Kstructured_choicesp >= 0 && len(va) > 1) {
			return( glob.Dats.ApStructuredChoice[ me.Kstructured_choicesp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "numeric_range" { // daml.unit:824, g_structh.act:609
		if (me.Knumeric_rangep >= 0 && len(va) > 1) {
			return( glob.Dats.ApNumericRange[ me.Knumeric_rangep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:170, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApArchitectureParam[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:819, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnhancedArchitectureParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnhancedArchitectureParam > daml.unit:819, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnhancedArchitectureParam > daml.unit:819, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnhancedArchitectureParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:170, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApArchitectureParam[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "structured_choices" {
		if me.Kstructured_choicesp >= 0 {
			st := glob.Dats.ApStructuredChoice[ me.Kstructured_choicesp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "numeric_range" {
		if me.Knumeric_rangep >= 0 {
			st := glob.Dats.ApNumericRange[ me.Knumeric_rangep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for EnhancedArchitectureParam %s,%s > daml.unit:819, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpKernelExecutionContext struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpKernelExecutionContext) TypeName() string {
    return me.Comp
}
func (me KpKernelExecutionContext) GetLineNo() string {
	return me.LineNo
}

func loadKernelExecutionContext(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpKernelExecutionContext)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApKernelExecutionContext)
	st.LineNo = lno
	st.Comp = "KernelExecutionContext";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["kernel_context_id"]
	act.index["KernelExecutionContext_" + name] = st.Me;
	st.MyName = name
	act.ApKernelExecutionContext = append(act.ApKernelExecutionContext, st)
	return 0
}

func (me KpKernelExecutionContext) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:834, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernelExecutionContext[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,KernelExecutionContext > daml.unit:834, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,KernelExecutionContext > daml.unit:834, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernelExecutionContext) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for KernelExecutionContext %s,%s > daml.unit:834, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProjectValidation struct {
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

func (me KpProjectValidation) TypeName() string {
    return me.Comp
}
func (me KpProjectValidation) GetLineNo() string {
	return me.LineNo
}

func loadProjectValidation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpProjectValidation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProjectValidation)
	st.LineNo = lno
	st.Comp = "ProjectValidation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ProjectValidation has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ProjectValidation under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsProjectValidation = append(act.ApProject[ len( act.ApProject )-1 ].ItsProjectValidation, st)	// daml.unit:623, g_structh.act:403
	name,_ := st.Names["validation_id"]
	s := strconv.Itoa(st.Kparentp) + "_ProjectValidation_" + name	// daml.unit:854, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApProjectValidation = append(act.ApProjectValidation, st)
	return 0
}

func (me KpProjectValidation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:623, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:850, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProjectValidation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ProjectValidation > daml.unit:850, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ProjectValidation > daml.unit:850, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProjectValidation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:623, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ProjectValidation %s,%s > daml.unit:850, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnumerationCatalog struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpEnumerationCatalog) TypeName() string {
    return me.Comp
}
func (me KpEnumerationCatalog) GetLineNo() string {
	return me.LineNo
}

func loadEnumerationCatalog(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEnumerationCatalog)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnumerationCatalog)
	st.LineNo = lno
	st.Comp = "EnumerationCatalog";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["enum_id"]
	act.index["EnumerationCatalog_" + name] = st.Me;
	st.MyName = name
	act.ApEnumerationCatalog = append(act.ApEnumerationCatalog, st)
	return 0
}

func (me KpEnumerationCatalog) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:866, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnumerationCatalog[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnumerationCatalog > daml.unit:866, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnumerationCatalog > daml.unit:866, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnumerationCatalog) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for EnumerationCatalog %s,%s > daml.unit:866, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCodeBlock struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	ItsClassCode [] *KpClassCode 
	ItsFunctionCode [] *KpFunctionCode 
	ItsMethodCode [] *KpMethodCode 
	Childs [] Kp
}

func (me KpCodeBlock) TypeName() string {
    return me.Comp
}
func (me KpCodeBlock) GetLineNo() string {
	return me.LineNo
}

func loadCodeBlock(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCodeBlock)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCodeBlock)
	st.LineNo = lno
	st.Comp = "CodeBlock";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["code_id"]
	act.index["CodeBlock_" + name] = st.Me;
	st.MyName = name
	act.ApCodeBlock = append(act.ApCodeBlock, st)
	return 0
}

func (me KpCodeBlock) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "SearchSpaceCode_operations_dict" && len(va) > 1) { // daml.unit:166, g_structh.act:698
		for _, st := range glob.Dats.ApSearchSpaceCode {
			if (st.Koperations_dictp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchSpaceCode_cell_code" && len(va) > 1) { // daml.unit:167, g_structh.act:698
		for _, st := range glob.Dats.ApSearchSpaceCode {
			if (st.Kcell_codep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Layer_code_block" && len(va) > 1) { // daml.unit:487, g_structh.act:698
		for _, st := range glob.Dats.ApLayer {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Op_code_block" && len(va) > 1) { // daml.unit:504, g_structh.act:698
		for _, st := range glob.Dats.ApOp {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OperationDef_code_block" && len(va) > 1) { // daml.unit:946, g_structh.act:698
		for _, st := range glob.Dats.ApOperationDef {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchMethod_code_block" && len(va) > 1) { // daml.unit:992, g_structh.act:698
		for _, st := range glob.Dats.ApSearchMethod {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "TrainingConfig_code_block" && len(va) > 1) { // daml.unit:1009, g_structh.act:698
		for _, st := range glob.Dats.ApTrainingConfig {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Architecture_main_code" && len(va) > 1) { // daml.unit:1035, g_structh.act:698
		for _, st := range glob.Dats.ApArchitecture {
			if (st.Kmain_codep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Architecture_config_code" && len(va) > 1) { // daml.unit:1036, g_structh.act:698
		for _, st := range glob.Dats.ApArchitecture {
			if (st.Kconfig_codep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "CodeDependency_code_block" && len(va) > 1) { // daml.unit:1065, g_structh.act:698
		for _, st := range glob.Dats.ApCodeDependency {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "CodeBlockReference_code_block" && len(va) > 1) { // daml.unit:1095, g_structh.act:698
		for _, st := range glob.Dats.ApCodeBlockReference {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "ActorTemplate_code_block" && len(va) > 1) { // daml.unit:1109, g_structh.act:698
		for _, st := range glob.Dats.ApActorTemplate {
			if (st.Kcode_blockp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:886, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CodeBlock > daml.unit:886, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CodeBlock > daml.unit:886, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCodeBlock) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ClassCode" { // daml.unit:899, g_structh.act:676
		for _, st := range me.ItsClassCode {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "FunctionCode" { // daml.unit:912, g_structh.act:676
		for _, st := range me.ItsFunctionCode {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "MethodCode" { // daml.unit:923, g_structh.act:676
		for _, st := range me.ItsMethodCode {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "SearchSpaceCode_operations_dict") { // daml.unit:166, g_structh.act:583
		for _, st := range glob.Dats.ApSearchSpaceCode {
			if (st.Koperations_dictp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "SearchSpaceCode_cell_code") { // daml.unit:167, g_structh.act:583
		for _, st := range glob.Dats.ApSearchSpaceCode {
			if (st.Kcell_codep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Layer_code_block") { // daml.unit:487, g_structh.act:583
		for _, st := range glob.Dats.ApLayer {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Op_code_block") { // daml.unit:504, g_structh.act:583
		for _, st := range glob.Dats.ApOp {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OperationDef_code_block") { // daml.unit:946, g_structh.act:583
		for _, st := range glob.Dats.ApOperationDef {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "SearchMethod_code_block") { // daml.unit:992, g_structh.act:583
		for _, st := range glob.Dats.ApSearchMethod {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "TrainingConfig_code_block") { // daml.unit:1009, g_structh.act:583
		for _, st := range glob.Dats.ApTrainingConfig {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Architecture_main_code") { // daml.unit:1035, g_structh.act:583
		for _, st := range glob.Dats.ApArchitecture {
			if (st.Kmain_codep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Architecture_config_code") { // daml.unit:1036, g_structh.act:583
		for _, st := range glob.Dats.ApArchitecture {
			if (st.Kconfig_codep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "CodeDependency_code_block") { // daml.unit:1065, g_structh.act:583
		for _, st := range glob.Dats.ApCodeDependency {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "CodeBlockReference_code_block") { // daml.unit:1095, g_structh.act:583
		for _, st := range glob.Dats.ApCodeBlockReference {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "ActorTemplate_code_block") { // daml.unit:1109, g_structh.act:583
		for _, st := range glob.Dats.ApActorTemplate {
			if (st.Kcode_blockp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for CodeBlock %s,%s > daml.unit:886, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpClassCode struct {
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

func (me KpClassCode) TypeName() string {
    return me.Comp
}
func (me KpClassCode) GetLineNo() string {
	return me.LineNo
}

func loadClassCode(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpClassCode)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApClassCode)
	st.LineNo = lno
	st.Comp = "ClassCode";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApCodeBlock ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ClassCode has no CodeBlock parent\n") ;
		return 1
	}
	st.Parent = act.ApCodeBlock[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ClassCode under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].Childs = append(act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].Childs, st)
	act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].ItsClassCode = append(act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].ItsClassCode, st)	// daml.unit:886, g_structh.act:403
	name,_ := st.Names["class_name"]
	s := strconv.Itoa(st.Kparentp) + "_ClassCode_" + name	// daml.unit:903, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApClassCode = append(act.ApClassCode, st)
	return 0
}

func (me KpClassCode) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:886, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "MethodCode_parent_class" && len(va) > 1) { // daml.unit:928, g_structh.act:698
		for _, st := range glob.Dats.ApMethodCode {
			if (st.Kparent_classp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // daml.unit:899, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApClassCode[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ClassCode > daml.unit:899, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ClassCode > daml.unit:899, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpClassCode) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:886, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "MethodCode_parent_class") { // daml.unit:928, g_structh.act:583
		for _, st := range glob.Dats.ApMethodCode {
			if (st.Kparent_classp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for ClassCode %s,%s > daml.unit:899, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFunctionCode struct {
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

func (me KpFunctionCode) TypeName() string {
    return me.Comp
}
func (me KpFunctionCode) GetLineNo() string {
	return me.LineNo
}

func loadFunctionCode(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFunctionCode)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFunctionCode)
	st.LineNo = lno
	st.Comp = "FunctionCode";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApCodeBlock ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FunctionCode has no CodeBlock parent\n") ;
		return 1
	}
	st.Parent = act.ApCodeBlock[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FunctionCode under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].Childs = append(act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].Childs, st)
	act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].ItsFunctionCode = append(act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].ItsFunctionCode, st)	// daml.unit:886, g_structh.act:403
	name,_ := st.Names["function_name"]
	s := strconv.Itoa(st.Kparentp) + "_FunctionCode_" + name	// daml.unit:916, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApFunctionCode = append(act.ApFunctionCode, st)
	return 0
}

func (me KpFunctionCode) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:886, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:912, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFunctionCode[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FunctionCode > daml.unit:912, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FunctionCode > daml.unit:912, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFunctionCode) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:886, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for FunctionCode %s,%s > daml.unit:912, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMethodCode struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kparent_classp int
}

func (me KpMethodCode) TypeName() string {
    return me.Comp
}
func (me KpMethodCode) GetLineNo() string {
	return me.LineNo
}

func loadMethodCode(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMethodCode)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMethodCode)
	st.LineNo = lno
	st.Comp = "MethodCode";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparent_classp = -1
	st.Kparentp = len( act.ApCodeBlock ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MethodCode has no CodeBlock parent\n") ;
		return 1
	}
	st.Parent = act.ApCodeBlock[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MethodCode under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].Childs = append(act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].Childs, st)
	act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].ItsMethodCode = append(act.ApCodeBlock[ len( act.ApCodeBlock )-1 ].ItsMethodCode, st)	// daml.unit:886, g_structh.act:403
	name,_ := st.Names["method_name"]
	s := strconv.Itoa(st.Kparentp) + "_MethodCode_" + name	// daml.unit:927, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApMethodCode = append(act.ApMethodCode, st)
	return 0
}

func (me KpMethodCode) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "parent_class" { // daml.unit:928, g_structh.act:609
		if (me.Kparent_classp >= 0 && len(va) > 1) {
			return( glob.Dats.ApClassCode[ me.Kparent_classp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // daml.unit:886, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:923, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMethodCode[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MethodCode > daml.unit:923, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MethodCode > daml.unit:923, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMethodCode) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:886, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "parent_class" {
		if me.Kparent_classp >= 0 {
			st := glob.Dats.ApClassCode[ me.Kparent_classp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MethodCode %s,%s > daml.unit:923, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOperationDef struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kcode_blockp int
	ItsOperationParam [] *KpOperationParam 
	Childs [] Kp
}

func (me KpOperationDef) TypeName() string {
    return me.Comp
}
func (me KpOperationDef) GetLineNo() string {
	return me.LineNo
}

func loadOperationDef(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOperationDef)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOperationDef)
	st.LineNo = lno
	st.Comp = "OperationDef";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcode_blockp = -1
	name,_ := st.Names["name"]
	act.index["OperationDef_" + name] = st.Me;
	st.MyName = name
	act.ApOperationDef = append(act.ApOperationDef, st)
	return 0
}

func (me KpOperationDef) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:946, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:939, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOperationDef[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OperationDef > daml.unit:939, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OperationDef > daml.unit:939, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOperationDef) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "OperationParam" { // daml.unit:951, g_structh.act:676
		for _, st := range me.ItsOperationParam {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for OperationDef %s,%s > daml.unit:939, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOperationParam struct {
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

func (me KpOperationParam) TypeName() string {
    return me.Comp
}
func (me KpOperationParam) GetLineNo() string {
	return me.LineNo
}

func loadOperationParam(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOperationParam)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOperationParam)
	st.LineNo = lno
	st.Comp = "OperationParam";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperationDef ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " OperationParam has no OperationDef parent\n") ;
		return 1
	}
	st.Parent = act.ApOperationDef[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " OperationParam under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperationDef[ len( act.ApOperationDef )-1 ].Childs = append(act.ApOperationDef[ len( act.ApOperationDef )-1 ].Childs, st)
	act.ApOperationDef[ len( act.ApOperationDef )-1 ].ItsOperationParam = append(act.ApOperationDef[ len( act.ApOperationDef )-1 ].ItsOperationParam, st)	// daml.unit:939, g_structh.act:403
	name,_ := st.Names["param_name"]
	s := strconv.Itoa(st.Kparentp) + "_OperationParam_" + name	// daml.unit:955, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOperationParam = append(act.ApOperationParam, st)
	return 0
}

func (me KpOperationParam) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // daml.unit:939, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperationDef[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // daml.unit:951, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOperationParam[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OperationParam > daml.unit:951, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OperationParam > daml.unit:951, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOperationParam) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // daml.unit:939, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperationDef[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for OperationParam %s,%s > daml.unit:951, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchMethod struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kcode_blockp int
}

func (me KpSearchMethod) TypeName() string {
    return me.Comp
}
func (me KpSearchMethod) GetLineNo() string {
	return me.LineNo
}

func loadSearchMethod(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchMethod)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchMethod)
	st.LineNo = lno
	st.Comp = "SearchMethod";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcode_blockp = -1
	name,_ := st.Names["method"]
	act.index["SearchMethod_" + name] = st.Me;
	st.MyName = name
	act.ApSearchMethod = append(act.ApSearchMethod, st)
	return 0
}

func (me KpSearchMethod) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:992, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:978, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchMethod[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchMethod > daml.unit:978, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchMethod > daml.unit:978, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchMethod) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchMethod %s,%s > daml.unit:978, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTrainingConfig struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kcode_blockp int
}

func (me KpTrainingConfig) TypeName() string {
    return me.Comp
}
func (me KpTrainingConfig) GetLineNo() string {
	return me.LineNo
}

func loadTrainingConfig(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTrainingConfig)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTrainingConfig)
	st.LineNo = lno
	st.Comp = "TrainingConfig";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcode_blockp = -1
	name,_ := st.Names["config_id"]
	act.index["TrainingConfig_" + name] = st.Me;
	st.MyName = name
	act.ApTrainingConfig = append(act.ApTrainingConfig, st)
	return 0
}

func (me KpTrainingConfig) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:1009, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:999, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTrainingConfig[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TrainingConfig > daml.unit:999, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TrainingConfig > daml.unit:999, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTrainingConfig) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TrainingConfig %s,%s > daml.unit:999, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpArchitecture struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kmain_codep int
	Kconfig_codep int
}

func (me KpArchitecture) TypeName() string {
    return me.Comp
}
func (me KpArchitecture) GetLineNo() string {
	return me.LineNo
}

func loadArchitecture(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpArchitecture)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApArchitecture)
	st.LineNo = lno
	st.Comp = "Architecture";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmain_codep = -1
	st.Kconfig_codep = -1
	name,_ := st.Names["name"]
	act.index["Architecture_" + name] = st.Me;
	st.MyName = name
	act.ApArchitecture = append(act.ApArchitecture, st)
	return 0
}

func (me KpArchitecture) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "main_code" { // daml.unit:1035, g_structh.act:609
		if (me.Kmain_codep >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kmain_codep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "config_code" { // daml.unit:1036, g_structh.act:609
		if (me.Kconfig_codep >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kconfig_codep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:1016, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApArchitecture[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Architecture > daml.unit:1016, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Architecture > daml.unit:1016, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpArchitecture) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "main_code" {
		if me.Kmain_codep >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kmain_codep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "config_code" {
		if me.Kconfig_codep >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kconfig_codep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Architecture %s,%s > daml.unit:1016, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTemplatePlaceholder struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpTemplatePlaceholder) TypeName() string {
    return me.Comp
}
func (me KpTemplatePlaceholder) GetLineNo() string {
	return me.LineNo
}

func loadTemplatePlaceholder(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTemplatePlaceholder)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTemplatePlaceholder)
	st.LineNo = lno
	st.Comp = "TemplatePlaceholder";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["placeholder_id"]
	act.index["TemplatePlaceholder_" + name] = st.Me;
	st.MyName = name
	act.ApTemplatePlaceholder = append(act.ApTemplatePlaceholder, st)
	return 0
}

func (me KpTemplatePlaceholder) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:1048, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTemplatePlaceholder[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TemplatePlaceholder > daml.unit:1048, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TemplatePlaceholder > daml.unit:1048, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTemplatePlaceholder) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for TemplatePlaceholder %s,%s > daml.unit:1048, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCodeDependency struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kcode_blockp int
}

func (me KpCodeDependency) TypeName() string {
    return me.Comp
}
func (me KpCodeDependency) GetLineNo() string {
	return me.LineNo
}

func loadCodeDependency(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCodeDependency)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCodeDependency)
	st.LineNo = lno
	st.Comp = "CodeDependency";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcode_blockp = -1
	name,_ := st.Names["dependency_id"]
	act.index["CodeDependency_" + name] = st.Me;
	st.MyName = name
	act.ApCodeDependency = append(act.ApCodeDependency, st)
	return 0
}

func (me KpCodeDependency) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:1065, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:1060, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCodeDependency[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CodeDependency > daml.unit:1060, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CodeDependency > daml.unit:1060, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCodeDependency) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CodeDependency %s,%s > daml.unit:1060, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCodegenTemplate struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpCodegenTemplate) TypeName() string {
    return me.Comp
}
func (me KpCodegenTemplate) GetLineNo() string {
	return me.LineNo
}

func loadCodegenTemplate(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCodegenTemplate)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCodegenTemplate)
	st.LineNo = lno
	st.Comp = "CodegenTemplate";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["template_id"]
	act.index["CodegenTemplate_" + name] = st.Me;
	st.MyName = name
	act.ApCodegenTemplate = append(act.ApCodegenTemplate, st)
	return 0
}

func (me KpCodegenTemplate) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // daml.unit:1075, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCodegenTemplate[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CodegenTemplate > daml.unit:1075, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CodegenTemplate > daml.unit:1075, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCodegenTemplate) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for CodegenTemplate %s,%s > daml.unit:1075, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCodeBlockReference struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kcode_blockp int
}

func (me KpCodeBlockReference) TypeName() string {
    return me.Comp
}
func (me KpCodeBlockReference) GetLineNo() string {
	return me.LineNo
}

func loadCodeBlockReference(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCodeBlockReference)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCodeBlockReference)
	st.LineNo = lno
	st.Comp = "CodeBlockReference";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcode_blockp = -1
	name,_ := st.Names["ref_id"]
	act.index["CodeBlockReference_" + name] = st.Me;
	st.MyName = name
	act.ApCodeBlockReference = append(act.ApCodeBlockReference, st)
	return 0
}

func (me KpCodeBlockReference) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:1095, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:1088, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlockReference[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CodeBlockReference > daml.unit:1088, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CodeBlockReference > daml.unit:1088, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCodeBlockReference) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CodeBlockReference %s,%s > daml.unit:1088, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpActorTemplate struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kcode_blockp int
}

func (me KpActorTemplate) TypeName() string {
    return me.Comp
}
func (me KpActorTemplate) GetLineNo() string {
	return me.LineNo
}

func loadActorTemplate(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpActorTemplate)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApActorTemplate)
	st.LineNo = lno
	st.Comp = "ActorTemplate";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcode_blockp = -1
	name,_ := st.Names["actor_name"]
	act.index["ActorTemplate_" + name] = st.Me;
	st.MyName = name
	act.ApActorTemplate = append(act.ApActorTemplate, st)
	return 0
}

func (me KpActorTemplate) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "code_block" { // daml.unit:1109, g_structh.act:609
		if (me.Kcode_blockp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCodeBlock[ me.Kcode_blockp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // daml.unit:1103, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApActorTemplate[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ActorTemplate > daml.unit:1103, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ActorTemplate > daml.unit:1103, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpActorTemplate) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "code_block" {
		if me.Kcode_blockp >= 0 {
			st := glob.Dats.ApCodeBlock[ me.Kcode_blockp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ActorTemplate %s,%s > daml.unit:1103, g_structh.act:209?", va[0], lno, me.LineNo)
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

