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

type KpProject struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	ItsComputeGraph [] *KpComputeGraph 
	ItsProvenance [] *KpProvenance 
	ItsOptimizationRun [] *KpOptimizationRun 
	ItsValidation [] *KpValidation 
	ItsGenotype [] *KpGenotype 
	ItsPhenotype [] *KpPhenotype 
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
	name,_ := st.Names["project"]
	act.index["Project_" + name] = st.Me;
	st.MyName = name
	act.ApProject = append(act.ApProject, st)
	return 0
}

func (me KpProject) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "ComputeGraph_project_ref" && len(va) > 1) { // nexus.unit:29, g_structh.act:698
		for _, st := range glob.Dats.ApComputeGraph {
			if (st.Kproject_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OptimizationRun_project_ref" && len(va) > 1) { // nexus.unit:420, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kproject_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OptimizationStrategy_project_ref" && len(va) > 1) { // nexus.unit:659, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationStrategy {
			if (st.Kproject_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Simulator_project_ref" && len(va) > 1) { // nexus.unit:717, g_structh.act:698
		for _, st := range glob.Dats.ApSimulator {
			if (st.Kproject_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Checkpoint_project_ref" && len(va) > 1) { // nexus.unit:793, g_structh.act:698
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kproject_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Checkpoint_project_source" && len(va) > 1) { // nexus.unit:798, g_structh.act:698
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kproject_sourcep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:12, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Project > nexus.unit:12, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Project > nexus.unit:12, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProject) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "ComputeGraph" { // nexus.unit:21, g_structh.act:676
		for _, st := range me.ItsComputeGraph {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Provenance" { // nexus.unit:397, g_structh.act:676
		for _, st := range me.ItsProvenance {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "OptimizationRun" { // nexus.unit:408, g_structh.act:676
		for _, st := range me.ItsOptimizationRun {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Validation" { // nexus.unit:424, g_structh.act:676
		for _, st := range me.ItsValidation {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Genotype" { // nexus.unit:435, g_structh.act:676
		for _, st := range me.ItsGenotype {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Phenotype" { // nexus.unit:446, g_structh.act:676
		for _, st := range me.ItsPhenotype {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "ComputeGraph_project_ref") { // nexus.unit:29, g_structh.act:583
		for _, st := range glob.Dats.ApComputeGraph {
			if (st.Kproject_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OptimizationRun_project_ref") { // nexus.unit:420, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kproject_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OptimizationStrategy_project_ref") { // nexus.unit:659, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationStrategy {
			if (st.Kproject_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Simulator_project_ref") { // nexus.unit:717, g_structh.act:583
		for _, st := range glob.Dats.ApSimulator {
			if (st.Kproject_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Checkpoint_project_ref") { // nexus.unit:793, g_structh.act:583
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kproject_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Checkpoint_project_source") { // nexus.unit:798, g_structh.act:583
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kproject_sourcep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for Project %s,%s > nexus.unit:12, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpComputeGraph struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kproject_refp int
	ItsHardwareTarget [] *KpHardwareTarget 
	ItsDataTensor [] *KpDataTensor 
	ItsOperation [] *KpOperation 
	ItsSearchSpace [] *KpSearchSpace 
	ItsEnergyBudget [] *KpEnergyBudget 
	ItsTileMapping [] *KpTileMapping 
	ItsPrivacyBudget [] *KpPrivacyBudget 
	ItsTraceCollection [] *KpTraceCollection 
	ItsEvolvableGraph [] *KpEvolvableGraph 
	Childs [] Kp
}

func (me KpComputeGraph) TypeName() string {
    return me.Comp
}
func (me KpComputeGraph) GetLineNo() string {
	return me.LineNo
}

func loadComputeGraph(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpComputeGraph)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApComputeGraph)
	st.LineNo = lno
	st.Comp = "ComputeGraph";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kproject_refp = -1
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ComputeGraph has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ComputeGraph under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsComputeGraph = append(act.ApProject[ len( act.ApProject )-1 ].ItsComputeGraph, st)	// nexus.unit:12, g_structh.act:403
	name,_ := st.Names["graph"]
	s := strconv.Itoa(st.Kparentp) + "_ComputeGraph_" + name	// nexus.unit:25, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApComputeGraph = append(act.ApComputeGraph, st)
	return 0
}

func (me KpComputeGraph) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project_ref" { // nexus.unit:29, g_structh.act:609
		if (me.Kproject_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kproject_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:12, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "TensorConsumer_graph" && len(va) > 1) { // nexus.unit:55, g_structh.act:698
		for _, st := range glob.Dats.ApTensorConsumer {
			if (st.Kgraphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OperationArg_graph" && len(va) > 1) { // nexus.unit:78, g_structh.act:698
		for _, st := range glob.Dats.ApOperationArg {
			if (st.Kgraphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OpDependency_graph" && len(va) > 1) { // nexus.unit:91, g_structh.act:698
		for _, st := range glob.Dats.ApOpDependency {
			if (st.Kgraphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "SearchTarget_graph" && len(va) > 1) { // nexus.unit:302, g_structh.act:698
		for _, st := range glob.Dats.ApSearchTarget {
			if (st.Kgraphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "EnergyAllocation_graph" && len(va) > 1) { // nexus.unit:337, g_structh.act:698
		for _, st := range glob.Dats.ApEnergyAllocation {
			if (st.Kgraphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "TileTarget_graph" && len(va) > 1) { // nexus.unit:359, g_structh.act:698
		for _, st := range glob.Dats.ApTileTarget {
			if (st.Kgraphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Phenotype_realized_graph" && len(va) > 1) { // nexus.unit:455, g_structh.act:698
		for _, st := range glob.Dats.ApPhenotype {
			if (st.Krealized_graphp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:21, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ComputeGraph > nexus.unit:21, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ComputeGraph > nexus.unit:21, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpComputeGraph) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "HardwareTarget" { // nexus.unit:31, g_structh.act:676
		for _, st := range me.ItsHardwareTarget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "DataTensor" { // nexus.unit:40, g_structh.act:676
		for _, st := range me.ItsDataTensor {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Operation" { // nexus.unit:60, g_structh.act:676
		for _, st := range me.ItsOperation {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "SearchSpace" { // nexus.unit:289, g_structh.act:676
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
	if va[0] == "EnergyBudget" { // nexus.unit:322, g_structh.act:676
		for _, st := range me.ItsEnergyBudget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "TileMapping" { // nexus.unit:344, g_structh.act:676
		for _, st := range me.ItsTileMapping {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "PrivacyBudget" { // nexus.unit:365, g_structh.act:676
		for _, st := range me.ItsPrivacyBudget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "TraceCollection" { // nexus.unit:374, g_structh.act:676
		for _, st := range me.ItsTraceCollection {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "EvolvableGraph" { // nexus.unit:384, g_structh.act:676
		for _, st := range me.ItsEvolvableGraph {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:12, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "project_ref" {
		if me.Kproject_refp >= 0 {
			st := glob.Dats.ApProject[ me.Kproject_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "TensorConsumer_graph") { // nexus.unit:55, g_structh.act:583
		for _, st := range glob.Dats.ApTensorConsumer {
			if (st.Kgraphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OperationArg_graph") { // nexus.unit:78, g_structh.act:583
		for _, st := range glob.Dats.ApOperationArg {
			if (st.Kgraphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OpDependency_graph") { // nexus.unit:91, g_structh.act:583
		for _, st := range glob.Dats.ApOpDependency {
			if (st.Kgraphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "SearchTarget_graph") { // nexus.unit:302, g_structh.act:583
		for _, st := range glob.Dats.ApSearchTarget {
			if (st.Kgraphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "EnergyAllocation_graph") { // nexus.unit:337, g_structh.act:583
		for _, st := range glob.Dats.ApEnergyAllocation {
			if (st.Kgraphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "TileTarget_graph") { // nexus.unit:359, g_structh.act:583
		for _, st := range glob.Dats.ApTileTarget {
			if (st.Kgraphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Phenotype_realized_graph") { // nexus.unit:455, g_structh.act:583
		for _, st := range glob.Dats.ApPhenotype {
			if (st.Krealized_graphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OptimizationStrategy_target_graph") { // nexus.unit:660, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationStrategy {
			if (st.Ktarget_graphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Simulator_target_graph") { // nexus.unit:718, g_structh.act:583
		for _, st := range glob.Dats.ApSimulator {
			if (st.Ktarget_graphp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Checkpoint_graph_snapshot") { // nexus.unit:799, g_structh.act:583
		for _, st := range glob.Dats.ApCheckpoint {
			if (st.Kgraph_snapshotp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for ComputeGraph %s,%s > nexus.unit:21, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHardwareTarget struct {
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
}

func (me KpHardwareTarget) TypeName() string {
    return me.Comp
}
func (me KpHardwareTarget) GetLineNo() string {
	return me.LineNo
}

func loadHardwareTarget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpHardwareTarget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHardwareTarget)
	st.LineNo = lno
	st.Comp = "HardwareTarget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " HardwareTarget has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " HardwareTarget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsHardwareTarget = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsHardwareTarget, st)	// nexus.unit:21, g_structh.act:403
	act.ApHardwareTarget = append(act.ApHardwareTarget, st)
	return 0
}

func (me KpHardwareTarget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // nexus.unit:35, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:31, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHardwareTarget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,HardwareTarget > nexus.unit:31, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,HardwareTarget > nexus.unit:31, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHardwareTarget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
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
	        fmt.Printf("?No its %s for HardwareTarget %s,%s > nexus.unit:31, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDataTensor struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsTensorConsumer [] *KpTensorConsumer 
	Childs [] Kp
}

func (me KpDataTensor) TypeName() string {
    return me.Comp
}
func (me KpDataTensor) GetLineNo() string {
	return me.LineNo
}

func loadDataTensor(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDataTensor)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDataTensor)
	st.LineNo = lno
	st.Comp = "DataTensor";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DataTensor has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DataTensor under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsDataTensor = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsDataTensor, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["tensor"]
	s := strconv.Itoa(st.Kparentp) + "_DataTensor_" + name	// nexus.unit:44, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApDataTensor = append(act.ApDataTensor, st)
	return 0
}

func (me KpDataTensor) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:40, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDataTensor[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DataTensor > nexus.unit:40, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DataTensor > nexus.unit:40, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDataTensor) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TensorConsumer" { // nexus.unit:51, g_structh.act:676
		for _, st := range me.ItsTensorConsumer {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "OperationArg_tensor_ref") { // nexus.unit:79, g_structh.act:583
		for _, st := range glob.Dats.ApOperationArg {
			if (st.Ktensor_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for DataTensor %s,%s > nexus.unit:40, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTensorConsumer struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kgraphp int
	Kconsumer_opp int
}

func (me KpTensorConsumer) TypeName() string {
    return me.Comp
}
func (me KpTensorConsumer) GetLineNo() string {
	return me.LineNo
}

func loadTensorConsumer(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTensorConsumer)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTensorConsumer)
	st.LineNo = lno
	st.Comp = "TensorConsumer";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kgraphp = -1
	st.Kconsumer_opp = -1
	st.Kparentp = len( act.ApDataTensor ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TensorConsumer has no DataTensor parent\n") ;
		return 1
	}
	st.Parent = act.ApDataTensor[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TensorConsumer under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApDataTensor[ len( act.ApDataTensor )-1 ].Childs = append(act.ApDataTensor[ len( act.ApDataTensor )-1 ].Childs, st)
	act.ApDataTensor[ len( act.ApDataTensor )-1 ].ItsTensorConsumer = append(act.ApDataTensor[ len( act.ApDataTensor )-1 ].ItsTensorConsumer, st)	// nexus.unit:40, g_structh.act:403
	act.ApTensorConsumer = append(act.ApTensorConsumer, st)
	return 0
}

func (me KpTensorConsumer) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "graph" { // nexus.unit:55, g_structh.act:609
		if (me.Kgraphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "consumer_op" { // nexus.unit:56, g_structh.act:609
		if (me.Kconsumer_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kconsumer_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:40, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDataTensor[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:51, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTensorConsumer[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TensorConsumer > nexus.unit:51, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TensorConsumer > nexus.unit:51, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTensorConsumer) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:40, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApDataTensor[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph" {
		if me.Kgraphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "consumer_op" {
		if me.Kconsumer_opp >= 0 {
			st := glob.Dats.ApOperation[ me.Kconsumer_opp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TensorConsumer %s,%s > nexus.unit:51, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOperation struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Khardware_targetp int
	Kkernel_refp int
	ItsOperationArg [] *KpOperationArg 
	ItsOpDependency [] *KpOpDependency 
	ItsClassicalOp [] *KpClassicalOp 
	ItsSpikingOp [] *KpSpikingOp 
	ItsAnalogOp [] *KpAnalogOp 
	ItsQuantumOp [] *KpQuantumOp 
	ItsPhotonicOp [] *KpPhotonicOp 
	ItsMolecularOp [] *KpMolecularOp 
	ItsHybridOp [] *KpHybridOp 
	ItsRedundancyStrategy [] *KpRedundancyStrategy 
	ItsProfilingHook [] *KpProfilingHook 
	ItsAdaptiveParameter [] *KpAdaptiveParameter 
	ItsSecureCompute [] *KpSecureCompute 
	Childs [] Kp
}

func (me KpOperation) TypeName() string {
    return me.Comp
}
func (me KpOperation) GetLineNo() string {
	return me.LineNo
}

func loadOperation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOperation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOperation)
	st.LineNo = lno
	st.Comp = "Operation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardware_targetp = -1
	st.Kkernel_refp = -1
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Operation has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Operation under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsOperation = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsOperation, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["op"]
	s := strconv.Itoa(st.Kparentp) + "_Operation_" + name	// nexus.unit:64, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOperation = append(act.ApOperation, st)
	return 0
}

func (me KpOperation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware_target" { // nexus.unit:69, g_structh.act:609
		if (me.Khardware_targetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardware_targetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "kernel_ref" { // nexus.unit:70, g_structh.act:609
		if (me.Kkernel_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kkernel_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "SimulationLevel_target_component" && len(va) > 1) { // nexus.unit:731, g_structh.act:698
		for _, st := range glob.Dats.ApSimulationLevel {
			if (st.Ktarget_componentp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:60, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Operation > nexus.unit:60, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Operation > nexus.unit:60, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOperation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "OperationArg" { // nexus.unit:72, g_structh.act:676
		for _, st := range me.ItsOperationArg {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "OpDependency" { // nexus.unit:84, g_structh.act:676
		for _, st := range me.ItsOpDependency {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "ClassicalOp" { // nexus.unit:94, g_structh.act:676
		for _, st := range me.ItsClassicalOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "SpikingOp" { // nexus.unit:105, g_structh.act:676
		for _, st := range me.ItsSpikingOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "AnalogOp" { // nexus.unit:132, g_structh.act:676
		for _, st := range me.ItsAnalogOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "QuantumOp" { // nexus.unit:146, g_structh.act:676
		for _, st := range me.ItsQuantumOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "PhotonicOp" { // nexus.unit:184, g_structh.act:676
		for _, st := range me.ItsPhotonicOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "MolecularOp" { // nexus.unit:197, g_structh.act:676
		for _, st := range me.ItsMolecularOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "HybridOp" { // nexus.unit:227, g_structh.act:676
		for _, st := range me.ItsHybridOp {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "RedundancyStrategy" { // nexus.unit:246, g_structh.act:676
		for _, st := range me.ItsRedundancyStrategy {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "ProfilingHook" { // nexus.unit:257, g_structh.act:676
		for _, st := range me.ItsProfilingHook {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "AdaptiveParameter" { // nexus.unit:268, g_structh.act:676
		for _, st := range me.ItsAdaptiveParameter {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "SecureCompute" { // nexus.unit:280, g_structh.act:676
		for _, st := range me.ItsSecureCompute {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "hardware_target" {
		if me.Khardware_targetp >= 0 {
			st := glob.Dats.ApHardware[ me.Khardware_targetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "kernel_ref" {
		if me.Kkernel_refp >= 0 {
			st := glob.Dats.ApKernel[ me.Kkernel_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "TensorConsumer_consumer_op") { // nexus.unit:56, g_structh.act:583
		for _, st := range glob.Dats.ApTensorConsumer {
			if (st.Kconsumer_opp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OpDependency_pred_op") { // nexus.unit:92, g_structh.act:583
		for _, st := range glob.Dats.ApOpDependency {
			if (st.Kpred_opp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "SearchTarget_operation") { // nexus.unit:303, g_structh.act:583
		for _, st := range glob.Dats.ApSearchTarget {
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
	if (va[0] == "EnergyAllocation_operation") { // nexus.unit:338, g_structh.act:583
		for _, st := range glob.Dats.ApEnergyAllocation {
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
	if (va[0] == "TileTarget_operation") { // nexus.unit:360, g_structh.act:583
		for _, st := range glob.Dats.ApTileTarget {
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
	if (va[0] == "SimulationLevel_target_component") { // nexus.unit:731, g_structh.act:583
		for _, st := range glob.Dats.ApSimulationLevel {
			if (st.Ktarget_componentp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for Operation %s,%s > nexus.unit:60, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOperationArg struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kgraphp int
	Ktensor_refp int
}

func (me KpOperationArg) TypeName() string {
    return me.Comp
}
func (me KpOperationArg) GetLineNo() string {
	return me.LineNo
}

func loadOperationArg(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOperationArg)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOperationArg)
	st.LineNo = lno
	st.Comp = "OperationArg";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kgraphp = -1
	st.Ktensor_refp = -1
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " OperationArg has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " OperationArg under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsOperationArg = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsOperationArg, st)	// nexus.unit:60, g_structh.act:403
	name,_ := st.Names["arg"]
	s := strconv.Itoa(st.Kparentp) + "_OperationArg_" + name	// nexus.unit:76, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOperationArg = append(act.ApOperationArg, st)
	return 0
}

func (me KpOperationArg) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "graph" { // nexus.unit:78, g_structh.act:609
		if (me.Kgraphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "tensor_ref" { // nexus.unit:79, g_structh.act:609
		if (me.Ktensor_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApDataTensor[ me.Ktensor_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:72, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOperationArg[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OperationArg > nexus.unit:72, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OperationArg > nexus.unit:72, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOperationArg) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph" {
		if me.Kgraphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "tensor_ref" {
		if me.Ktensor_refp >= 0 {
			st := glob.Dats.ApDataTensor[ me.Ktensor_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for OperationArg %s,%s > nexus.unit:72, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOpDependency struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kgraphp int
	Kpred_opp int
}

func (me KpOpDependency) TypeName() string {
    return me.Comp
}
func (me KpOpDependency) GetLineNo() string {
	return me.LineNo
}

func loadOpDependency(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOpDependency)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOpDependency)
	st.LineNo = lno
	st.Comp = "OpDependency";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kgraphp = -1
	st.Kpred_opp = -1
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " OpDependency has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " OpDependency under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsOpDependency = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsOpDependency, st)	// nexus.unit:60, g_structh.act:403
	act.ApOpDependency = append(act.ApOpDependency, st)
	return 0
}

func (me KpOpDependency) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "graph" { // nexus.unit:91, g_structh.act:609
		if (me.Kgraphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "pred_op" { // nexus.unit:92, g_structh.act:609
		if (me.Kpred_opp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kpred_opp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:84, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOpDependency[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OpDependency > nexus.unit:84, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OpDependency > nexus.unit:84, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOpDependency) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph" {
		if me.Kgraphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "pred_op" {
		if me.Kpred_opp >= 0 {
			st := glob.Dats.ApOperation[ me.Kpred_opp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for OpDependency %s,%s > nexus.unit:84, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpClassicalOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kfusion_patternp int
}

func (me KpClassicalOp) TypeName() string {
    return me.Comp
}
func (me KpClassicalOp) GetLineNo() string {
	return me.LineNo
}

func loadClassicalOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpClassicalOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApClassicalOp)
	st.LineNo = lno
	st.Comp = "ClassicalOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kfusion_patternp = -1
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ClassicalOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ClassicalOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsClassicalOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsClassicalOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApClassicalOp = append(act.ApClassicalOp, st)
	return 0
}

func (me KpClassicalOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "fusion_pattern" { // nexus.unit:103, g_structh.act:609
		if (me.Kfusion_patternp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFusionPattern[ me.Kfusion_patternp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:94, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApClassicalOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ClassicalOp > nexus.unit:94, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ClassicalOp > nexus.unit:94, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpClassicalOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "fusion_pattern" {
		if me.Kfusion_patternp >= 0 {
			st := glob.Dats.ApFusionPattern[ me.Kfusion_patternp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ClassicalOp %s,%s > nexus.unit:94, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSpikingOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsPlasticityRule [] *KpPlasticityRule 
	Childs [] Kp
}

func (me KpSpikingOp) TypeName() string {
    return me.Comp
}
func (me KpSpikingOp) GetLineNo() string {
	return me.LineNo
}

func loadSpikingOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSpikingOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSpikingOp)
	st.LineNo = lno
	st.Comp = "SpikingOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SpikingOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SpikingOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsSpikingOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsSpikingOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApSpikingOp = append(act.ApSpikingOp, st)
	return 0
}

func (me KpSpikingOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:105, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSpikingOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SpikingOp > nexus.unit:105, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SpikingOp > nexus.unit:105, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSpikingOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "PlasticityRule" { // nexus.unit:118, g_structh.act:676
		for _, st := range me.ItsPlasticityRule {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SpikingOp %s,%s > nexus.unit:105, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPlasticityRule struct {
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

func (me KpPlasticityRule) TypeName() string {
    return me.Comp
}
func (me KpPlasticityRule) GetLineNo() string {
	return me.LineNo
}

func loadPlasticityRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPlasticityRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPlasticityRule)
	st.LineNo = lno
	st.Comp = "PlasticityRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSpikingOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PlasticityRule has no SpikingOp parent\n") ;
		return 1
	}
	st.Parent = act.ApSpikingOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PlasticityRule under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSpikingOp[ len( act.ApSpikingOp )-1 ].Childs = append(act.ApSpikingOp[ len( act.ApSpikingOp )-1 ].Childs, st)
	act.ApSpikingOp[ len( act.ApSpikingOp )-1 ].ItsPlasticityRule = append(act.ApSpikingOp[ len( act.ApSpikingOp )-1 ].ItsPlasticityRule, st)	// nexus.unit:105, g_structh.act:403
	act.ApPlasticityRule = append(act.ApPlasticityRule, st)
	return 0
}

func (me KpPlasticityRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:105, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSpikingOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:118, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPlasticityRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PlasticityRule > nexus.unit:118, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PlasticityRule > nexus.unit:118, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPlasticityRule) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:105, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSpikingOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PlasticityRule %s,%s > nexus.unit:118, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAnalogOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Karray_targetp int
}

func (me KpAnalogOp) TypeName() string {
    return me.Comp
}
func (me KpAnalogOp) GetLineNo() string {
	return me.LineNo
}

func loadAnalogOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAnalogOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAnalogOp)
	st.LineNo = lno
	st.Comp = "AnalogOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Karray_targetp = -1
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AnalogOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AnalogOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsAnalogOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsAnalogOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApAnalogOp = append(act.ApAnalogOp, st)
	return 0
}

func (me KpAnalogOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "array_target" { // nexus.unit:144, g_structh.act:609
		if (me.Karray_targetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApAnalogHardware[ me.Karray_targetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:132, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAnalogOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AnalogOp > nexus.unit:132, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AnalogOp > nexus.unit:132, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAnalogOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "array_target" {
		if me.Karray_targetp >= 0 {
			st := glob.Dats.ApAnalogHardware[ me.Karray_targetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for AnalogOp %s,%s > nexus.unit:132, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpQuantumOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsQubitTarget [] *KpQubitTarget 
	ItsControlQubit [] *KpControlQubit 
	ItsQuantumCircuit [] *KpQuantumCircuit 
	Childs [] Kp
}

func (me KpQuantumOp) TypeName() string {
    return me.Comp
}
func (me KpQuantumOp) GetLineNo() string {
	return me.LineNo
}

func loadQuantumOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpQuantumOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApQuantumOp)
	st.LineNo = lno
	st.Comp = "QuantumOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " QuantumOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " QuantumOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsQuantumOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsQuantumOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApQuantumOp = append(act.ApQuantumOp, st)
	return 0
}

func (me KpQuantumOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:146, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,QuantumOp > nexus.unit:146, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,QuantumOp > nexus.unit:146, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpQuantumOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "QubitTarget" { // nexus.unit:157, g_structh.act:676
		for _, st := range me.ItsQubitTarget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "ControlQubit" { // nexus.unit:165, g_structh.act:676
		for _, st := range me.ItsControlQubit {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "QuantumCircuit" { // nexus.unit:173, g_structh.act:676
		for _, st := range me.ItsQuantumCircuit {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for QuantumOp %s,%s > nexus.unit:146, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpQubitTarget struct {
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

func (me KpQubitTarget) TypeName() string {
    return me.Comp
}
func (me KpQubitTarget) GetLineNo() string {
	return me.LineNo
}

func loadQubitTarget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpQubitTarget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApQubitTarget)
	st.LineNo = lno
	st.Comp = "QubitTarget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApQuantumOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " QubitTarget has no QuantumOp parent\n") ;
		return 1
	}
	st.Parent = act.ApQuantumOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " QubitTarget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].Childs = append(act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].Childs, st)
	act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].ItsQubitTarget = append(act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].ItsQubitTarget, st)	// nexus.unit:146, g_structh.act:403
	name,_ := st.Names["qubit_index"]
	s := strconv.Itoa(st.Kparentp) + "_QubitTarget_" + name	// nexus.unit:161, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApQubitTarget = append(act.ApQubitTarget, st)
	return 0
}

func (me KpQubitTarget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:146, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:157, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApQubitTarget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,QubitTarget > nexus.unit:157, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,QubitTarget > nexus.unit:157, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpQubitTarget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:146, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApQuantumOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for QubitTarget %s,%s > nexus.unit:157, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpControlQubit struct {
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

func (me KpControlQubit) TypeName() string {
    return me.Comp
}
func (me KpControlQubit) GetLineNo() string {
	return me.LineNo
}

func loadControlQubit(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpControlQubit)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApControlQubit)
	st.LineNo = lno
	st.Comp = "ControlQubit";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApQuantumOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ControlQubit has no QuantumOp parent\n") ;
		return 1
	}
	st.Parent = act.ApQuantumOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ControlQubit under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].Childs = append(act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].Childs, st)
	act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].ItsControlQubit = append(act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].ItsControlQubit, st)	// nexus.unit:146, g_structh.act:403
	name,_ := st.Names["qubit_index"]
	s := strconv.Itoa(st.Kparentp) + "_ControlQubit_" + name	// nexus.unit:169, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApControlQubit = append(act.ApControlQubit, st)
	return 0
}

func (me KpControlQubit) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:146, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:165, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApControlQubit[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ControlQubit > nexus.unit:165, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ControlQubit > nexus.unit:165, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpControlQubit) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:146, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApQuantumOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ControlQubit %s,%s > nexus.unit:165, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpQuantumCircuit struct {
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

func (me KpQuantumCircuit) TypeName() string {
    return me.Comp
}
func (me KpQuantumCircuit) GetLineNo() string {
	return me.LineNo
}

func loadQuantumCircuit(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpQuantumCircuit)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApQuantumCircuit)
	st.LineNo = lno
	st.Comp = "QuantumCircuit";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApQuantumOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " QuantumCircuit has no QuantumOp parent\n") ;
		return 1
	}
	st.Parent = act.ApQuantumOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " QuantumCircuit under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].Childs = append(act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].Childs, st)
	act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].ItsQuantumCircuit = append(act.ApQuantumOp[ len( act.ApQuantumOp )-1 ].ItsQuantumCircuit, st)	// nexus.unit:146, g_structh.act:403
	name,_ := st.Names["circuit_name"]
	s := strconv.Itoa(st.Kparentp) + "_QuantumCircuit_" + name	// nexus.unit:177, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApQuantumCircuit = append(act.ApQuantumCircuit, st)
	return 0
}

func (me KpQuantumCircuit) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:146, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:173, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumCircuit[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,QuantumCircuit > nexus.unit:173, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,QuantumCircuit > nexus.unit:173, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpQuantumCircuit) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:146, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApQuantumOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for QuantumCircuit %s,%s > nexus.unit:173, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPhotonicOp struct {
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

func (me KpPhotonicOp) TypeName() string {
    return me.Comp
}
func (me KpPhotonicOp) GetLineNo() string {
	return me.LineNo
}

func loadPhotonicOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPhotonicOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPhotonicOp)
	st.LineNo = lno
	st.Comp = "PhotonicOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PhotonicOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PhotonicOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsPhotonicOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsPhotonicOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApPhotonicOp = append(act.ApPhotonicOp, st)
	return 0
}

func (me KpPhotonicOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:184, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPhotonicOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PhotonicOp > nexus.unit:184, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PhotonicOp > nexus.unit:184, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPhotonicOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PhotonicOp %s,%s > nexus.unit:184, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMolecularOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsReactant [] *KpReactant 
	ItsProduct [] *KpProduct 
	Childs [] Kp
}

func (me KpMolecularOp) TypeName() string {
    return me.Comp
}
func (me KpMolecularOp) GetLineNo() string {
	return me.LineNo
}

func loadMolecularOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMolecularOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMolecularOp)
	st.LineNo = lno
	st.Comp = "MolecularOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MolecularOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MolecularOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsMolecularOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsMolecularOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApMolecularOp = append(act.ApMolecularOp, st)
	return 0
}

func (me KpMolecularOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:197, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMolecularOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MolecularOp > nexus.unit:197, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MolecularOp > nexus.unit:197, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMolecularOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Reactant" { // nexus.unit:209, g_structh.act:676
		for _, st := range me.ItsReactant {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Product" { // nexus.unit:218, g_structh.act:676
		for _, st := range me.ItsProduct {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MolecularOp %s,%s > nexus.unit:197, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpReactant struct {
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

func (me KpReactant) TypeName() string {
    return me.Comp
}
func (me KpReactant) GetLineNo() string {
	return me.LineNo
}

func loadReactant(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpReactant)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApReactant)
	st.LineNo = lno
	st.Comp = "Reactant";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApMolecularOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Reactant has no MolecularOp parent\n") ;
		return 1
	}
	st.Parent = act.ApMolecularOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Reactant under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].Childs = append(act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].Childs, st)
	act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].ItsReactant = append(act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].ItsReactant, st)	// nexus.unit:197, g_structh.act:403
	name,_ := st.Names["species"]
	s := strconv.Itoa(st.Kparentp) + "_Reactant_" + name	// nexus.unit:213, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApReactant = append(act.ApReactant, st)
	return 0
}

func (me KpReactant) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:197, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMolecularOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:209, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApReactant[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Reactant > nexus.unit:209, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Reactant > nexus.unit:209, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpReactant) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:197, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApMolecularOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Reactant %s,%s > nexus.unit:209, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProduct struct {
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

func (me KpProduct) TypeName() string {
    return me.Comp
}
func (me KpProduct) GetLineNo() string {
	return me.LineNo
}

func loadProduct(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpProduct)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProduct)
	st.LineNo = lno
	st.Comp = "Product";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApMolecularOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Product has no MolecularOp parent\n") ;
		return 1
	}
	st.Parent = act.ApMolecularOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Product under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].Childs = append(act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].Childs, st)
	act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].ItsProduct = append(act.ApMolecularOp[ len( act.ApMolecularOp )-1 ].ItsProduct, st)	// nexus.unit:197, g_structh.act:403
	name,_ := st.Names["species"]
	s := strconv.Itoa(st.Kparentp) + "_Product_" + name	// nexus.unit:222, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApProduct = append(act.ApProduct, st)
	return 0
}

func (me KpProduct) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:197, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMolecularOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:218, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProduct[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Product > nexus.unit:218, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Product > nexus.unit:218, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProduct) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:197, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApMolecularOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Product %s,%s > nexus.unit:218, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpHybridOp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsFallbackMode [] *KpFallbackMode 
	Childs [] Kp
}

func (me KpHybridOp) TypeName() string {
    return me.Comp
}
func (me KpHybridOp) GetLineNo() string {
	return me.LineNo
}

func loadHybridOp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpHybridOp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApHybridOp)
	st.LineNo = lno
	st.Comp = "HybridOp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " HybridOp has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " HybridOp under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsHybridOp = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsHybridOp, st)	// nexus.unit:60, g_structh.act:403
	act.ApHybridOp = append(act.ApHybridOp, st)
	return 0
}

func (me KpHybridOp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:227, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHybridOp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,HybridOp > nexus.unit:227, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,HybridOp > nexus.unit:227, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHybridOp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "FallbackMode" { // nexus.unit:237, g_structh.act:676
		for _, st := range me.ItsFallbackMode {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for HybridOp %s,%s > nexus.unit:227, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFallbackMode struct {
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

func (me KpFallbackMode) TypeName() string {
    return me.Comp
}
func (me KpFallbackMode) GetLineNo() string {
	return me.LineNo
}

func loadFallbackMode(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFallbackMode)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFallbackMode)
	st.LineNo = lno
	st.Comp = "FallbackMode";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHybridOp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FallbackMode has no HybridOp parent\n") ;
		return 1
	}
	st.Parent = act.ApHybridOp[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FallbackMode under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHybridOp[ len( act.ApHybridOp )-1 ].Childs = append(act.ApHybridOp[ len( act.ApHybridOp )-1 ].Childs, st)
	act.ApHybridOp[ len( act.ApHybridOp )-1 ].ItsFallbackMode = append(act.ApHybridOp[ len( act.ApHybridOp )-1 ].ItsFallbackMode, st)	// nexus.unit:227, g_structh.act:403
	act.ApFallbackMode = append(act.ApFallbackMode, st)
	return 0
}

func (me KpFallbackMode) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:227, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHybridOp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:237, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFallbackMode[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FallbackMode > nexus.unit:237, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FallbackMode > nexus.unit:237, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFallbackMode) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:227, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHybridOp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for FallbackMode %s,%s > nexus.unit:237, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpRedundancyStrategy struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kcheckpointsp int
}

func (me KpRedundancyStrategy) TypeName() string {
    return me.Comp
}
func (me KpRedundancyStrategy) GetLineNo() string {
	return me.LineNo
}

func loadRedundancyStrategy(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpRedundancyStrategy)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApRedundancyStrategy)
	st.LineNo = lno
	st.Comp = "RedundancyStrategy";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcheckpointsp = -1
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " RedundancyStrategy has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " RedundancyStrategy under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsRedundancyStrategy = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsRedundancyStrategy, st)	// nexus.unit:60, g_structh.act:403
	act.ApRedundancyStrategy = append(act.ApRedundancyStrategy, st)
	return 0
}

func (me KpRedundancyStrategy) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "checkpoints" { // nexus.unit:255, g_structh.act:609
		if (me.Kcheckpointsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCheckpoint[ me.Kcheckpointsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:246, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApRedundancyStrategy[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,RedundancyStrategy > nexus.unit:246, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,RedundancyStrategy > nexus.unit:246, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpRedundancyStrategy) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "checkpoints" {
		if me.Kcheckpointsp >= 0 {
			st := glob.Dats.ApCheckpoint[ me.Kcheckpointsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for RedundancyStrategy %s,%s > nexus.unit:246, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProfilingHook struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kmetrics_refp int
}

func (me KpProfilingHook) TypeName() string {
    return me.Comp
}
func (me KpProfilingHook) GetLineNo() string {
	return me.LineNo
}

func loadProfilingHook(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpProfilingHook)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProfilingHook)
	st.LineNo = lno
	st.Comp = "ProfilingHook";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmetrics_refp = -1
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ProfilingHook has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ProfilingHook under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsProfilingHook = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsProfilingHook, st)	// nexus.unit:60, g_structh.act:403
	name,_ := st.Names["hook_id"]
	s := strconv.Itoa(st.Kparentp) + "_ProfilingHook_" + name	// nexus.unit:261, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApProfilingHook = append(act.ApProfilingHook, st)
	return 0
}

func (me KpProfilingHook) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "metrics_ref" { // nexus.unit:266, g_structh.act:609
		if (me.Kmetrics_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMetric[ me.Kmetrics_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:257, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProfilingHook[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ProfilingHook > nexus.unit:257, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ProfilingHook > nexus.unit:257, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProfilingHook) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "metrics_ref" {
		if me.Kmetrics_refp >= 0 {
			st := glob.Dats.ApMetric[ me.Kmetrics_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ProfilingHook %s,%s > nexus.unit:257, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kparentp int
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
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AdaptiveParameter has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AdaptiveParameter under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsAdaptiveParameter = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsAdaptiveParameter, st)	// nexus.unit:60, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_AdaptiveParameter_" + name	// nexus.unit:272, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApAdaptiveParameter = append(act.ApAdaptiveParameter, st)
	return 0
}

func (me KpAdaptiveParameter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:268, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAdaptiveParameter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AdaptiveParameter > nexus.unit:268, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AdaptiveParameter > nexus.unit:268, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAdaptiveParameter) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for AdaptiveParameter %s,%s > nexus.unit:268, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSecureCompute struct {
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

func (me KpSecureCompute) TypeName() string {
    return me.Comp
}
func (me KpSecureCompute) GetLineNo() string {
	return me.LineNo
}

func loadSecureCompute(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSecureCompute)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSecureCompute)
	st.LineNo = lno
	st.Comp = "SecureCompute";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApOperation ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SecureCompute has no Operation parent\n") ;
		return 1
	}
	st.Parent = act.ApOperation[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SecureCompute under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOperation[ len( act.ApOperation )-1 ].Childs = append(act.ApOperation[ len( act.ApOperation )-1 ].Childs, st)
	act.ApOperation[ len( act.ApOperation )-1 ].ItsSecureCompute = append(act.ApOperation[ len( act.ApOperation )-1 ].ItsSecureCompute, st)	// nexus.unit:60, g_structh.act:403
	act.ApSecureCompute = append(act.ApSecureCompute, st)
	return 0
}

func (me KpSecureCompute) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:60, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:280, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSecureCompute[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SecureCompute > nexus.unit:280, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SecureCompute > nexus.unit:280, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSecureCompute) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:60, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOperation[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SecureCompute %s,%s > nexus.unit:280, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsSearchTarget [] *KpSearchTarget 
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
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchSpace has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchSpace under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsSearchSpace = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsSearchSpace, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["search_space"]
	s := strconv.Itoa(st.Kparentp) + "_SearchSpace_" + name	// nexus.unit:293, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchSpace = append(act.ApSearchSpace, st)
	return 0
}

func (me KpSearchSpace) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:289, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchSpace > nexus.unit:289, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchSpace > nexus.unit:289, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchSpace) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "SearchTarget" { // nexus.unit:298, g_structh.act:676
		for _, st := range me.ItsSearchTarget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "SearchParameter" { // nexus.unit:308, g_structh.act:676
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
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "OptimizationStrategy_search_space") { // nexus.unit:661, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationStrategy {
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
	        fmt.Printf("?No its %s for SearchSpace %s,%s > nexus.unit:289, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSearchTarget struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kgraphp int
	Koperationp int
}

func (me KpSearchTarget) TypeName() string {
    return me.Comp
}
func (me KpSearchTarget) GetLineNo() string {
	return me.LineNo
}

func loadSearchTarget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSearchTarget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSearchTarget)
	st.LineNo = lno
	st.Comp = "SearchTarget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kgraphp = -1
	st.Koperationp = -1
	st.Kparentp = len( act.ApSearchSpace ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SearchTarget has no SearchSpace parent\n") ;
		return 1
	}
	st.Parent = act.ApSearchSpace[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SearchTarget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].Childs, st)
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchTarget = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchTarget, st)	// nexus.unit:289, g_structh.act:403
	act.ApSearchTarget = append(act.ApSearchTarget, st)
	return 0
}

func (me KpSearchTarget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "graph" { // nexus.unit:302, g_structh.act:609
		if (me.Kgraphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "operation" { // nexus.unit:303, g_structh.act:609
		if (me.Koperationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Koperationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:289, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:298, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchTarget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchTarget > nexus.unit:298, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchTarget > nexus.unit:298, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchTarget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:289, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph" {
		if me.Kgraphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "operation" {
		if me.Koperationp >= 0 {
			st := glob.Dats.ApOperation[ me.Koperationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchTarget %s,%s > nexus.unit:298, g_structh.act:209?", va[0], lno, me.LineNo)
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
	act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchParameter = append(act.ApSearchSpace[ len( act.ApSearchSpace )-1 ].ItsSearchParameter, st)	// nexus.unit:289, g_structh.act:403
	name,_ := st.Names["param"]
	s := strconv.Itoa(st.Kparentp) + "_SearchParameter_" + name	// nexus.unit:312, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSearchParameter = append(act.ApSearchParameter, st)
	return 0
}

func (me KpSearchParameter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:289, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:308, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSearchParameter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SearchParameter > nexus.unit:308, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SearchParameter > nexus.unit:308, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSearchParameter) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:289, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSearchSpace[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SearchParameter %s,%s > nexus.unit:308, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnergyBudget struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsEnergyAllocation [] *KpEnergyAllocation 
	Childs [] Kp
}

func (me KpEnergyBudget) TypeName() string {
    return me.Comp
}
func (me KpEnergyBudget) GetLineNo() string {
	return me.LineNo
}

func loadEnergyBudget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEnergyBudget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnergyBudget)
	st.LineNo = lno
	st.Comp = "EnergyBudget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " EnergyBudget has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " EnergyBudget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsEnergyBudget = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsEnergyBudget, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["budget_id"]
	s := strconv.Itoa(st.Kparentp) + "_EnergyBudget_" + name	// nexus.unit:326, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApEnergyBudget = append(act.ApEnergyBudget, st)
	return 0
}

func (me KpEnergyBudget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:322, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyBudget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnergyBudget > nexus.unit:322, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnergyBudget > nexus.unit:322, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnergyBudget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "EnergyAllocation" { // nexus.unit:333, g_structh.act:676
		for _, st := range me.ItsEnergyAllocation {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for EnergyBudget %s,%s > nexus.unit:322, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEnergyAllocation struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kgraphp int
	Koperationp int
}

func (me KpEnergyAllocation) TypeName() string {
    return me.Comp
}
func (me KpEnergyAllocation) GetLineNo() string {
	return me.LineNo
}

func loadEnergyAllocation(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEnergyAllocation)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEnergyAllocation)
	st.LineNo = lno
	st.Comp = "EnergyAllocation";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kgraphp = -1
	st.Koperationp = -1
	st.Kparentp = len( act.ApEnergyBudget ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " EnergyAllocation has no EnergyBudget parent\n") ;
		return 1
	}
	st.Parent = act.ApEnergyBudget[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " EnergyAllocation under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApEnergyBudget[ len( act.ApEnergyBudget )-1 ].Childs = append(act.ApEnergyBudget[ len( act.ApEnergyBudget )-1 ].Childs, st)
	act.ApEnergyBudget[ len( act.ApEnergyBudget )-1 ].ItsEnergyAllocation = append(act.ApEnergyBudget[ len( act.ApEnergyBudget )-1 ].ItsEnergyAllocation, st)	// nexus.unit:322, g_structh.act:403
	act.ApEnergyAllocation = append(act.ApEnergyAllocation, st)
	return 0
}

func (me KpEnergyAllocation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "graph" { // nexus.unit:337, g_structh.act:609
		if (me.Kgraphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "operation" { // nexus.unit:338, g_structh.act:609
		if (me.Koperationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Koperationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:322, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyBudget[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:333, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEnergyAllocation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EnergyAllocation > nexus.unit:333, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EnergyAllocation > nexus.unit:333, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEnergyAllocation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:322, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApEnergyBudget[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph" {
		if me.Kgraphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "operation" {
		if me.Koperationp >= 0 {
			st := glob.Dats.ApOperation[ me.Koperationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for EnergyAllocation %s,%s > nexus.unit:333, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTileMapping struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	ItsTileTarget [] *KpTileTarget 
	Childs [] Kp
}

func (me KpTileMapping) TypeName() string {
    return me.Comp
}
func (me KpTileMapping) GetLineNo() string {
	return me.LineNo
}

func loadTileMapping(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTileMapping)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTileMapping)
	st.LineNo = lno
	st.Comp = "TileMapping";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TileMapping has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TileMapping under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsTileMapping = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsTileMapping, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["tile_id"]
	s := strconv.Itoa(st.Kparentp) + "_TileMapping_" + name	// nexus.unit:348, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTileMapping = append(act.ApTileMapping, st)
	return 0
}

func (me KpTileMapping) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:344, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTileMapping[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TileMapping > nexus.unit:344, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TileMapping > nexus.unit:344, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTileMapping) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "TileTarget" { // nexus.unit:355, g_structh.act:676
		for _, st := range me.ItsTileTarget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TileMapping %s,%s > nexus.unit:344, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTileTarget struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kgraphp int
	Koperationp int
}

func (me KpTileTarget) TypeName() string {
    return me.Comp
}
func (me KpTileTarget) GetLineNo() string {
	return me.LineNo
}

func loadTileTarget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTileTarget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTileTarget)
	st.LineNo = lno
	st.Comp = "TileTarget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kgraphp = -1
	st.Koperationp = -1
	st.Kparentp = len( act.ApTileMapping ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TileTarget has no TileMapping parent\n") ;
		return 1
	}
	st.Parent = act.ApTileMapping[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TileTarget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApTileMapping[ len( act.ApTileMapping )-1 ].Childs = append(act.ApTileMapping[ len( act.ApTileMapping )-1 ].Childs, st)
	act.ApTileMapping[ len( act.ApTileMapping )-1 ].ItsTileTarget = append(act.ApTileMapping[ len( act.ApTileMapping )-1 ].ItsTileTarget, st)	// nexus.unit:344, g_structh.act:403
	act.ApTileTarget = append(act.ApTileTarget, st)
	return 0
}

func (me KpTileTarget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "graph" { // nexus.unit:359, g_structh.act:609
		if (me.Kgraphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "operation" { // nexus.unit:360, g_structh.act:609
		if (me.Koperationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Koperationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:344, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApTileMapping[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:355, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTileTarget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TileTarget > nexus.unit:355, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TileTarget > nexus.unit:355, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTileTarget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:344, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApTileMapping[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph" {
		if me.Kgraphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "operation" {
		if me.Koperationp >= 0 {
			st := glob.Dats.ApOperation[ me.Koperationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TileTarget %s,%s > nexus.unit:355, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPrivacyBudget struct {
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

func (me KpPrivacyBudget) TypeName() string {
    return me.Comp
}
func (me KpPrivacyBudget) GetLineNo() string {
	return me.LineNo
}

func loadPrivacyBudget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPrivacyBudget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPrivacyBudget)
	st.LineNo = lno
	st.Comp = "PrivacyBudget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PrivacyBudget has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PrivacyBudget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsPrivacyBudget = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsPrivacyBudget, st)	// nexus.unit:21, g_structh.act:403
	act.ApPrivacyBudget = append(act.ApPrivacyBudget, st)
	return 0
}

func (me KpPrivacyBudget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:365, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPrivacyBudget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PrivacyBudget > nexus.unit:365, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PrivacyBudget > nexus.unit:365, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPrivacyBudget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PrivacyBudget %s,%s > nexus.unit:365, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpTraceCollection struct {
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

func (me KpTraceCollection) TypeName() string {
    return me.Comp
}
func (me KpTraceCollection) GetLineNo() string {
	return me.LineNo
}

func loadTraceCollection(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpTraceCollection)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApTraceCollection)
	st.LineNo = lno
	st.Comp = "TraceCollection";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " TraceCollection has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " TraceCollection under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsTraceCollection = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsTraceCollection, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["trace_id"]
	s := strconv.Itoa(st.Kparentp) + "_TraceCollection_" + name	// nexus.unit:378, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApTraceCollection = append(act.ApTraceCollection, st)
	return 0
}

func (me KpTraceCollection) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:374, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApTraceCollection[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,TraceCollection > nexus.unit:374, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,TraceCollection > nexus.unit:374, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpTraceCollection) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for TraceCollection %s,%s > nexus.unit:374, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpEvolvableGraph struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Kfitness_metricp int
}

func (me KpEvolvableGraph) TypeName() string {
    return me.Comp
}
func (me KpEvolvableGraph) GetLineNo() string {
	return me.LineNo
}

func loadEvolvableGraph(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpEvolvableGraph)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApEvolvableGraph)
	st.LineNo = lno
	st.Comp = "EvolvableGraph";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kfitness_metricp = -1
	st.Kparentp = len( act.ApComputeGraph ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " EvolvableGraph has no ComputeGraph parent\n") ;
		return 1
	}
	st.Parent = act.ApComputeGraph[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " EvolvableGraph under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].Childs, st)
	act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsEvolvableGraph = append(act.ApComputeGraph[ len( act.ApComputeGraph )-1 ].ItsEvolvableGraph, st)	// nexus.unit:21, g_structh.act:403
	name,_ := st.Names["evolvable_id"]
	s := strconv.Itoa(st.Kparentp) + "_EvolvableGraph_" + name	// nexus.unit:388, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApEvolvableGraph = append(act.ApEvolvableGraph, st)
	return 0
}

func (me KpEvolvableGraph) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "fitness_metric" { // nexus.unit:395, g_structh.act:609
		if (me.Kfitness_metricp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMetric[ me.Kfitness_metricp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:21, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:384, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApEvolvableGraph[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,EvolvableGraph > nexus.unit:384, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,EvolvableGraph > nexus.unit:384, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpEvolvableGraph) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:21, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "fitness_metric" {
		if me.Kfitness_metricp >= 0 {
			st := glob.Dats.ApMetric[ me.Kfitness_metricp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for EvolvableGraph %s,%s > nexus.unit:384, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpProvenance struct {
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

func (me KpProvenance) TypeName() string {
    return me.Comp
}
func (me KpProvenance) GetLineNo() string {
	return me.LineNo
}

func loadProvenance(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpProvenance)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApProvenance)
	st.LineNo = lno
	st.Comp = "Provenance";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Provenance has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Provenance under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsProvenance = append(act.ApProject[ len( act.ApProject )-1 ].ItsProvenance, st)	// nexus.unit:12, g_structh.act:403
	act.ApProvenance = append(act.ApProvenance, st)
	return 0
}

func (me KpProvenance) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:12, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:397, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApProvenance[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Provenance > nexus.unit:397, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Provenance > nexus.unit:397, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpProvenance) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:12, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Provenance %s,%s > nexus.unit:397, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kparentp int
	Kproject_refp int
	Kstrategy_refp int
	Kbest_checkpointp int
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
	st.Kproject_refp = -1
	st.Kstrategy_refp = -1
	st.Kbest_checkpointp = -1
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " OptimizationRun has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " OptimizationRun under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsOptimizationRun = append(act.ApProject[ len( act.ApProject )-1 ].ItsOptimizationRun, st)	// nexus.unit:12, g_structh.act:403
	name,_ := st.Names["run_id"]
	s := strconv.Itoa(st.Kparentp) + "_OptimizationRun_" + name	// nexus.unit:412, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOptimizationRun = append(act.ApOptimizationRun, st)
	return 0
}

func (me KpOptimizationRun) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project_ref" { // nexus.unit:420, g_structh.act:609
		if (me.Kproject_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kproject_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "strategy_ref" { // nexus.unit:421, g_structh.act:609
		if (me.Kstrategy_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationStrategy[ me.Kstrategy_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "best_checkpoint" { // nexus.unit:422, g_structh.act:609
		if (me.Kbest_checkpointp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCheckpoint[ me.Kbest_checkpointp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:12, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:408, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationRun[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OptimizationRun > nexus.unit:408, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OptimizationRun > nexus.unit:408, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOptimizationRun) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:12, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "project_ref" {
		if me.Kproject_refp >= 0 {
			st := glob.Dats.ApProject[ me.Kproject_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "strategy_ref" {
		if me.Kstrategy_refp >= 0 {
			st := glob.Dats.ApOptimizationStrategy[ me.Kstrategy_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "best_checkpoint" {
		if me.Kbest_checkpointp >= 0 {
			st := glob.Dats.ApCheckpoint[ me.Kbest_checkpointp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for OptimizationRun %s,%s > nexus.unit:408, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kparentp int
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
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Validation has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Validation under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsValidation = append(act.ApProject[ len( act.ApProject )-1 ].ItsValidation, st)	// nexus.unit:12, g_structh.act:403
	name,_ := st.Names["rule"]
	s := strconv.Itoa(st.Kparentp) + "_Validation_" + name	// nexus.unit:428, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApValidation = append(act.ApValidation, st)
	return 0
}

func (me KpValidation) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:12, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:424, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApValidation[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Validation > nexus.unit:424, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Validation > nexus.unit:424, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpValidation) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:12, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Validation %s,%s > nexus.unit:424, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpGenotype struct {
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

func (me KpGenotype) TypeName() string {
    return me.Comp
}
func (me KpGenotype) GetLineNo() string {
	return me.LineNo
}

func loadGenotype(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpGenotype)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApGenotype)
	st.LineNo = lno
	st.Comp = "Genotype";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Genotype has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Genotype under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsGenotype = append(act.ApProject[ len( act.ApProject )-1 ].ItsGenotype, st)	// nexus.unit:12, g_structh.act:403
	name,_ := st.Names["genome_id"]
	s := strconv.Itoa(st.Kparentp) + "_Genotype_" + name	// nexus.unit:439, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApGenotype = append(act.ApGenotype, st)
	return 0
}

func (me KpGenotype) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:12, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "Phenotype_genome_source" && len(va) > 1) { // nexus.unit:456, g_structh.act:698
		for _, st := range glob.Dats.ApPhenotype {
			if (st.Kgenome_sourcep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:435, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApGenotype[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Genotype > nexus.unit:435, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Genotype > nexus.unit:435, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpGenotype) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:12, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Phenotype_genome_source") { // nexus.unit:456, g_structh.act:583
		for _, st := range glob.Dats.ApPhenotype {
			if (st.Kgenome_sourcep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for Genotype %s,%s > nexus.unit:435, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPhenotype struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Krealized_graphp int
	Kgenome_sourcep int
}

func (me KpPhenotype) TypeName() string {
    return me.Comp
}
func (me KpPhenotype) GetLineNo() string {
	return me.LineNo
}

func loadPhenotype(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPhenotype)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPhenotype)
	st.LineNo = lno
	st.Comp = "Phenotype";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Krealized_graphp = -1
	st.Kgenome_sourcep = -1
	st.Kparentp = len( act.ApProject ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Phenotype has no Project parent\n") ;
		return 1
	}
	st.Parent = act.ApProject[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " Phenotype under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApProject[ len( act.ApProject )-1 ].Childs = append(act.ApProject[ len( act.ApProject )-1 ].Childs, st)
	act.ApProject[ len( act.ApProject )-1 ].ItsPhenotype = append(act.ApProject[ len( act.ApProject )-1 ].ItsPhenotype, st)	// nexus.unit:12, g_structh.act:403
	name,_ := st.Names["phenotype_id"]
	s := strconv.Itoa(st.Kparentp) + "_Phenotype_" + name	// nexus.unit:450, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApPhenotype = append(act.ApPhenotype, st)
	return 0
}

func (me KpPhenotype) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "realized_graph" { // nexus.unit:455, g_structh.act:609
		if (me.Krealized_graphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Krealized_graphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "genome_source" { // nexus.unit:456, g_structh.act:609
		if (me.Kgenome_sourcep >= 0 && len(va) > 1) {
			return( glob.Dats.ApGenotype[ me.Kgenome_sourcep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:12, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:446, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPhenotype[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Phenotype > nexus.unit:446, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Phenotype > nexus.unit:446, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPhenotype) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:12, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApProject[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "realized_graph" {
		if me.Krealized_graphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Krealized_graphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "genome_source" {
		if me.Kgenome_sourcep >= 0 {
			st := glob.Dats.ApGenotype[ me.Kgenome_sourcep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Phenotype %s,%s > nexus.unit:446, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsPhysicsModel [] *KpPhysicsModel 
	ItsClassicalHardware [] *KpClassicalHardware 
	ItsNeuromorphicHardware [] *KpNeuromorphicHardware 
	ItsAnalogHardware [] *KpAnalogHardware 
	ItsQuantumHardware [] *KpQuantumHardware 
	ItsPhotonicHardware [] *KpPhotonicHardware 
	ItsMolecularHardware [] *KpMolecularHardware 
	ItsPowerDomain [] *KpPowerDomain 
	ItsSpatialArray [] *KpSpatialArray 
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
	if (va[0] == "HardwareTarget_hardware" && len(va) > 1) { // nexus.unit:35, g_structh.act:698
		for _, st := range glob.Dats.ApHardwareTarget {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Operation_hardware_target" && len(va) > 1) { // nexus.unit:69, g_structh.act:698
		for _, st := range glob.Dats.ApOperation {
			if (st.Khardware_targetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Kernel_hardware_target" && len(va) > 1) { // nexus.unit:597, g_structh.act:698
		for _, st := range glob.Dats.ApKernel {
			if (st.Khardware_targetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "FusionHardwareTarget_hardware" && len(va) > 1) { // nexus.unit:615, g_structh.act:698
		for _, st := range glob.Dats.ApFusionHardwareTarget {
			if (st.Khardwarep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:462, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Hardware > nexus.unit:462, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Hardware > nexus.unit:462, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "PhysicsModel" { // nexus.unit:473, g_structh.act:676
		for _, st := range me.ItsPhysicsModel {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "ClassicalHardware" { // nexus.unit:486, g_structh.act:676
		for _, st := range me.ItsClassicalHardware {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "NeuromorphicHardware" { // nexus.unit:497, g_structh.act:676
		for _, st := range me.ItsNeuromorphicHardware {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "AnalogHardware" { // nexus.unit:509, g_structh.act:676
		for _, st := range me.ItsAnalogHardware {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "QuantumHardware" { // nexus.unit:522, g_structh.act:676
		for _, st := range me.ItsQuantumHardware {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "PhotonicHardware" { // nexus.unit:536, g_structh.act:676
		for _, st := range me.ItsPhotonicHardware {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "MolecularHardware" { // nexus.unit:549, g_structh.act:676
		for _, st := range me.ItsMolecularHardware {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "PowerDomain" { // nexus.unit:561, g_structh.act:676
		for _, st := range me.ItsPowerDomain {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "SpatialArray" { // nexus.unit:571, g_structh.act:676
		for _, st := range me.ItsSpatialArray {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "HardwareTarget_hardware") { // nexus.unit:35, g_structh.act:583
		for _, st := range glob.Dats.ApHardwareTarget {
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
	if (va[0] == "Operation_hardware_target") { // nexus.unit:69, g_structh.act:583
		for _, st := range glob.Dats.ApOperation {
			if (st.Khardware_targetp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Kernel_hardware_target") { // nexus.unit:597, g_structh.act:583
		for _, st := range glob.Dats.ApKernel {
			if (st.Khardware_targetp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "FusionHardwareTarget_hardware") { // nexus.unit:615, g_structh.act:583
		for _, st := range glob.Dats.ApFusionHardwareTarget {
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
	        fmt.Printf("?No its %s for Hardware %s,%s > nexus.unit:462, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPhysicsModel struct {
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

func (me KpPhysicsModel) TypeName() string {
    return me.Comp
}
func (me KpPhysicsModel) GetLineNo() string {
	return me.LineNo
}

func loadPhysicsModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPhysicsModel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPhysicsModel)
	st.LineNo = lno
	st.Comp = "PhysicsModel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PhysicsModel has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PhysicsModel under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsPhysicsModel = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsPhysicsModel, st)	// nexus.unit:462, g_structh.act:403
	name,_ := st.Names["model_id"]
	s := strconv.Itoa(st.Kparentp) + "_PhysicsModel_" + name	// nexus.unit:477, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApPhysicsModel = append(act.ApPhysicsModel, st)
	return 0
}

func (me KpPhysicsModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:473, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPhysicsModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PhysicsModel > nexus.unit:473, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PhysicsModel > nexus.unit:473, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPhysicsModel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PhysicsModel %s,%s > nexus.unit:473, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpClassicalHardware struct {
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

func (me KpClassicalHardware) TypeName() string {
    return me.Comp
}
func (me KpClassicalHardware) GetLineNo() string {
	return me.LineNo
}

func loadClassicalHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpClassicalHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApClassicalHardware)
	st.LineNo = lno
	st.Comp = "ClassicalHardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " ClassicalHardware has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " ClassicalHardware under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsClassicalHardware = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsClassicalHardware, st)	// nexus.unit:462, g_structh.act:403
	act.ApClassicalHardware = append(act.ApClassicalHardware, st)
	return 0
}

func (me KpClassicalHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:486, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApClassicalHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ClassicalHardware > nexus.unit:486, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ClassicalHardware > nexus.unit:486, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpClassicalHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for ClassicalHardware %s,%s > nexus.unit:486, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuromorphicHardware struct {
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

func (me KpNeuromorphicHardware) TypeName() string {
    return me.Comp
}
func (me KpNeuromorphicHardware) GetLineNo() string {
	return me.LineNo
}

func loadNeuromorphicHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuromorphicHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuromorphicHardware)
	st.LineNo = lno
	st.Comp = "NeuromorphicHardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " NeuromorphicHardware has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " NeuromorphicHardware under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsNeuromorphicHardware = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsNeuromorphicHardware, st)	// nexus.unit:462, g_structh.act:403
	act.ApNeuromorphicHardware = append(act.ApNeuromorphicHardware, st)
	return 0
}

func (me KpNeuromorphicHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:497, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuromorphicHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuromorphicHardware > nexus.unit:497, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuromorphicHardware > nexus.unit:497, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuromorphicHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for NeuromorphicHardware %s,%s > nexus.unit:497, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAnalogHardware struct {
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

func (me KpAnalogHardware) TypeName() string {
    return me.Comp
}
func (me KpAnalogHardware) GetLineNo() string {
	return me.LineNo
}

func loadAnalogHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAnalogHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAnalogHardware)
	st.LineNo = lno
	st.Comp = "AnalogHardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AnalogHardware has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AnalogHardware under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsAnalogHardware = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsAnalogHardware, st)	// nexus.unit:462, g_structh.act:403
	act.ApAnalogHardware = append(act.ApAnalogHardware, st)
	return 0
}

func (me KpAnalogHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "AnalogOp_array_target" && len(va) > 1) { // nexus.unit:144, g_structh.act:698
		for _, st := range glob.Dats.ApAnalogOp {
			if (st.Karray_targetp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:509, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAnalogHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AnalogHardware > nexus.unit:509, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AnalogHardware > nexus.unit:509, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAnalogHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "AnalogOp_array_target") { // nexus.unit:144, g_structh.act:583
		for _, st := range glob.Dats.ApAnalogOp {
			if (st.Karray_targetp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for AnalogHardware %s,%s > nexus.unit:509, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpQuantumHardware struct {
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

func (me KpQuantumHardware) TypeName() string {
    return me.Comp
}
func (me KpQuantumHardware) GetLineNo() string {
	return me.LineNo
}

func loadQuantumHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpQuantumHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApQuantumHardware)
	st.LineNo = lno
	st.Comp = "QuantumHardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " QuantumHardware has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " QuantumHardware under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsQuantumHardware = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsQuantumHardware, st)	// nexus.unit:462, g_structh.act:403
	act.ApQuantumHardware = append(act.ApQuantumHardware, st)
	return 0
}

func (me KpQuantumHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:522, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,QuantumHardware > nexus.unit:522, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,QuantumHardware > nexus.unit:522, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpQuantumHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for QuantumHardware %s,%s > nexus.unit:522, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPhotonicHardware struct {
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

func (me KpPhotonicHardware) TypeName() string {
    return me.Comp
}
func (me KpPhotonicHardware) GetLineNo() string {
	return me.LineNo
}

func loadPhotonicHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPhotonicHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPhotonicHardware)
	st.LineNo = lno
	st.Comp = "PhotonicHardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PhotonicHardware has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PhotonicHardware under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsPhotonicHardware = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsPhotonicHardware, st)	// nexus.unit:462, g_structh.act:403
	act.ApPhotonicHardware = append(act.ApPhotonicHardware, st)
	return 0
}

func (me KpPhotonicHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:536, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPhotonicHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PhotonicHardware > nexus.unit:536, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PhotonicHardware > nexus.unit:536, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPhotonicHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PhotonicHardware %s,%s > nexus.unit:536, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMolecularHardware struct {
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

func (me KpMolecularHardware) TypeName() string {
    return me.Comp
}
func (me KpMolecularHardware) GetLineNo() string {
	return me.LineNo
}

func loadMolecularHardware(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpMolecularHardware)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMolecularHardware)
	st.LineNo = lno
	st.Comp = "MolecularHardware";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " MolecularHardware has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " MolecularHardware under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsMolecularHardware = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsMolecularHardware, st)	// nexus.unit:462, g_structh.act:403
	act.ApMolecularHardware = append(act.ApMolecularHardware, st)
	return 0
}

func (me KpMolecularHardware) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:549, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMolecularHardware[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,MolecularHardware > nexus.unit:549, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,MolecularHardware > nexus.unit:549, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMolecularHardware) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for MolecularHardware %s,%s > nexus.unit:549, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpPowerDomain struct {
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

func (me KpPowerDomain) TypeName() string {
    return me.Comp
}
func (me KpPowerDomain) GetLineNo() string {
	return me.LineNo
}

func loadPowerDomain(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpPowerDomain)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApPowerDomain)
	st.LineNo = lno
	st.Comp = "PowerDomain";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " PowerDomain has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " PowerDomain under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsPowerDomain = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsPowerDomain, st)	// nexus.unit:462, g_structh.act:403
	name,_ := st.Names["domain_id"]
	s := strconv.Itoa(st.Kparentp) + "_PowerDomain_" + name	// nexus.unit:565, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApPowerDomain = append(act.ApPowerDomain, st)
	return 0
}

func (me KpPowerDomain) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:561, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApPowerDomain[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,PowerDomain > nexus.unit:561, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,PowerDomain > nexus.unit:561, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpPowerDomain) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for PowerDomain %s,%s > nexus.unit:561, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSpatialArray struct {
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

func (me KpSpatialArray) TypeName() string {
    return me.Comp
}
func (me KpSpatialArray) GetLineNo() string {
	return me.LineNo
}

func loadSpatialArray(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSpatialArray)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSpatialArray)
	st.LineNo = lno
	st.Comp = "SpatialArray";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApHardware ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SpatialArray has no Hardware parent\n") ;
		return 1
	}
	st.Parent = act.ApHardware[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SpatialArray under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApHardware[ len( act.ApHardware )-1 ].Childs = append(act.ApHardware[ len( act.ApHardware )-1 ].Childs, st)
	act.ApHardware[ len( act.ApHardware )-1 ].ItsSpatialArray = append(act.ApHardware[ len( act.ApHardware )-1 ].ItsSpatialArray, st)	// nexus.unit:462, g_structh.act:403
	name,_ := st.Names["array_id"]
	s := strconv.Itoa(st.Kparentp) + "_SpatialArray_" + name	// nexus.unit:575, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSpatialArray = append(act.ApSpatialArray, st)
	return 0
}

func (me KpSpatialArray) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:462, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:571, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSpatialArray[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SpatialArray > nexus.unit:571, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SpatialArray > nexus.unit:571, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSpatialArray) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:462, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApHardware[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SpatialArray %s,%s > nexus.unit:571, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Khardware_targetp int
	Kfusion_sourcep int
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
	st.Khardware_targetp = -1
	st.Kfusion_sourcep = -1
	name,_ := st.Names["kernel"]
	act.index["Kernel_" + name] = st.Me;
	st.MyName = name
	act.ApKernel = append(act.ApKernel, st)
	return 0
}

func (me KpKernel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware_target" { // nexus.unit:597, g_structh.act:609
		if (me.Khardware_targetp >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardware_targetp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "fusion_source" { // nexus.unit:598, g_structh.act:609
		if (me.Kfusion_sourcep >= 0 && len(va) > 1) {
			return( glob.Dats.ApFusionPattern[ me.Kfusion_sourcep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Operation_kernel_ref" && len(va) > 1) { // nexus.unit:70, g_structh.act:698
		for _, st := range glob.Dats.ApOperation {
			if (st.Kkernel_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "FusionPattern_fused_kernel_ref" && len(va) > 1) { // nexus.unit:609, g_structh.act:698
		for _, st := range glob.Dats.ApFusionPattern {
			if (st.Kfused_kernel_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:585, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Kernel > nexus.unit:585, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Kernel > nexus.unit:585, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpKernel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "hardware_target" {
		if me.Khardware_targetp >= 0 {
			st := glob.Dats.ApHardware[ me.Khardware_targetp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "fusion_source" {
		if me.Kfusion_sourcep >= 0 {
			st := glob.Dats.ApFusionPattern[ me.Kfusion_sourcep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Operation_kernel_ref") { // nexus.unit:70, g_structh.act:583
		for _, st := range glob.Dats.ApOperation {
			if (st.Kkernel_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "FusionPattern_fused_kernel_ref") { // nexus.unit:609, g_structh.act:583
		for _, st := range glob.Dats.ApFusionPattern {
			if (st.Kfused_kernel_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for Kernel %s,%s > nexus.unit:585, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFusionPattern struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kfused_kernel_refp int
	ItsFusionHardwareTarget [] *KpFusionHardwareTarget 
	ItsFusionOpTarget [] *KpFusionOpTarget 
	Childs [] Kp
}

func (me KpFusionPattern) TypeName() string {
    return me.Comp
}
func (me KpFusionPattern) GetLineNo() string {
	return me.LineNo
}

func loadFusionPattern(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFusionPattern)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFusionPattern)
	st.LineNo = lno
	st.Comp = "FusionPattern";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kfused_kernel_refp = -1
	name,_ := st.Names["pattern"]
	act.index["FusionPattern_" + name] = st.Me;
	st.MyName = name
	act.ApFusionPattern = append(act.ApFusionPattern, st)
	return 0
}

func (me KpFusionPattern) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "fused_kernel_ref" { // nexus.unit:609, g_structh.act:609
		if (me.Kfused_kernel_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApKernel[ me.Kfused_kernel_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "ClassicalOp_fusion_pattern" && len(va) > 1) { // nexus.unit:103, g_structh.act:698
		for _, st := range glob.Dats.ApClassicalOp {
			if (st.Kfusion_patternp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Kernel_fusion_source" && len(va) > 1) { // nexus.unit:598, g_structh.act:698
		for _, st := range glob.Dats.ApKernel {
			if (st.Kfusion_sourcep == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:600, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusionPattern[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FusionPattern > nexus.unit:600, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FusionPattern > nexus.unit:600, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFusionPattern) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "FusionHardwareTarget" { // nexus.unit:611, g_structh.act:676
		for _, st := range me.ItsFusionHardwareTarget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "FusionOpTarget" { // nexus.unit:619, g_structh.act:676
		for _, st := range me.ItsFusionOpTarget {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "fused_kernel_ref" {
		if me.Kfused_kernel_refp >= 0 {
			st := glob.Dats.ApKernel[ me.Kfused_kernel_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "ClassicalOp_fusion_pattern") { // nexus.unit:103, g_structh.act:583
		for _, st := range glob.Dats.ApClassicalOp {
			if (st.Kfusion_patternp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "Kernel_fusion_source") { // nexus.unit:598, g_structh.act:583
		for _, st := range glob.Dats.ApKernel {
			if (st.Kfusion_sourcep == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for FusionPattern %s,%s > nexus.unit:600, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFusionHardwareTarget struct {
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
}

func (me KpFusionHardwareTarget) TypeName() string {
    return me.Comp
}
func (me KpFusionHardwareTarget) GetLineNo() string {
	return me.LineNo
}

func loadFusionHardwareTarget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFusionHardwareTarget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFusionHardwareTarget)
	st.LineNo = lno
	st.Comp = "FusionHardwareTarget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Khardwarep = -1
	st.Kparentp = len( act.ApFusionPattern ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FusionHardwareTarget has no FusionPattern parent\n") ;
		return 1
	}
	st.Parent = act.ApFusionPattern[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FusionHardwareTarget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].Childs = append(act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].Childs, st)
	act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].ItsFusionHardwareTarget = append(act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].ItsFusionHardwareTarget, st)	// nexus.unit:600, g_structh.act:403
	act.ApFusionHardwareTarget = append(act.ApFusionHardwareTarget, st)
	return 0
}

func (me KpFusionHardwareTarget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "hardware" { // nexus.unit:615, g_structh.act:609
		if (me.Khardwarep >= 0 && len(va) > 1) {
			return( glob.Dats.ApHardware[ me.Khardwarep ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:600, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFusionPattern[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:611, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusionHardwareTarget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FusionHardwareTarget > nexus.unit:611, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FusionHardwareTarget > nexus.unit:611, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFusionHardwareTarget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:600, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApFusionPattern[ me.Kparentp ]
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
	        fmt.Printf("?No its %s for FusionHardwareTarget %s,%s > nexus.unit:611, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFusionOpTarget struct {
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

func (me KpFusionOpTarget) TypeName() string {
    return me.Comp
}
func (me KpFusionOpTarget) GetLineNo() string {
	return me.LineNo
}

func loadFusionOpTarget(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFusionOpTarget)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFusionOpTarget)
	st.LineNo = lno
	st.Comp = "FusionOpTarget";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApFusionPattern ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FusionOpTarget has no FusionPattern parent\n") ;
		return 1
	}
	st.Parent = act.ApFusionPattern[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FusionOpTarget under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].Childs = append(act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].Childs, st)
	act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].ItsFusionOpTarget = append(act.ApFusionPattern[ len( act.ApFusionPattern )-1 ].ItsFusionOpTarget, st)	// nexus.unit:600, g_structh.act:403
	name,_ := st.Names["op_type"]
	s := strconv.Itoa(st.Kparentp) + "_FusionOpTarget_" + name	// nexus.unit:623, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApFusionOpTarget = append(act.ApFusionOpTarget, st)
	return 0
}

func (me KpFusionOpTarget) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:600, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFusionPattern[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:619, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusionOpTarget[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FusionOpTarget > nexus.unit:619, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FusionOpTarget > nexus.unit:619, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFusionOpTarget) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:600, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApFusionPattern[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for FusionOpTarget %s,%s > nexus.unit:619, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDataFormatConverter struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDataFormatConverter) TypeName() string {
    return me.Comp
}
func (me KpDataFormatConverter) GetLineNo() string {
	return me.LineNo
}

func loadDataFormatConverter(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDataFormatConverter)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDataFormatConverter)
	st.LineNo = lno
	st.Comp = "DataFormatConverter";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["converter"]
	act.index["DataFormatConverter_" + name] = st.Me;
	st.MyName = name
	act.ApDataFormatConverter = append(act.ApDataFormatConverter, st)
	return 0
}

func (me KpDataFormatConverter) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:632, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDataFormatConverter[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DataFormatConverter > nexus.unit:632, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DataFormatConverter > nexus.unit:632, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDataFormatConverter) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DataFormatConverter %s,%s > nexus.unit:632, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOptimizationStrategy struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kproject_refp int
	Ktarget_graphp int
	Ksearch_spacep int
	Kstrategyp int
	ItsFitnessFunction [] *KpFitnessFunction 
	Childs [] Kp
}

func (me KpOptimizationStrategy) TypeName() string {
    return me.Comp
}
func (me KpOptimizationStrategy) GetLineNo() string {
	return me.LineNo
}

func loadOptimizationStrategy(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpOptimizationStrategy)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOptimizationStrategy)
	st.LineNo = lno
	st.Comp = "OptimizationStrategy";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kproject_refp = -1
	st.Ktarget_graphp = -1
	st.Ksearch_spacep = -1
	st.Kstrategyp = -1
	name,_ := st.Names["strategy"]
	act.index["OptimizationStrategy_" + name] = st.Me;
	st.MyName = name
	act.ApOptimizationStrategy = append(act.ApOptimizationStrategy, st)
	return 0
}

func (me KpOptimizationStrategy) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project_ref" { // nexus.unit:659, g_structh.act:609
		if (me.Kproject_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kproject_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "target_graph" { // nexus.unit:660, g_structh.act:609
		if (me.Ktarget_graphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Ktarget_graphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "search_space" { // nexus.unit:661, g_structh.act:609
		if (me.Ksearch_spacep >= 0 && len(va) > 1) {
			return( glob.Dats.ApSearchSpace[ me.Ksearch_spacep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "strategy" { // nexus.unit:662, g_structh.act:609
		if (me.Kstrategyp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationStrategy[ me.Kstrategyp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "OptimizationRun_strategy_ref" && len(va) > 1) { // nexus.unit:421, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kstrategy_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OptimizationStrategy_strategy" && len(va) > 1) { // nexus.unit:662, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationStrategy {
			if (st.Kstrategyp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:648, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationStrategy[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,OptimizationStrategy > nexus.unit:648, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,OptimizationStrategy > nexus.unit:648, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpOptimizationStrategy) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "FitnessFunction" { // nexus.unit:665, g_structh.act:676
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
	if va[0] == "project_ref" {
		if me.Kproject_refp >= 0 {
			st := glob.Dats.ApProject[ me.Kproject_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "target_graph" {
		if me.Ktarget_graphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Ktarget_graphp ]
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
			st := glob.Dats.ApOptimizationStrategy[ me.Kstrategyp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "OptimizationRun_strategy_ref") { // nexus.unit:421, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kstrategy_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OptimizationStrategy_strategy") { // nexus.unit:662, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationStrategy {
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
	        fmt.Printf("?No its %s for OptimizationStrategy %s,%s > nexus.unit:648, g_structh.act:209?", va[0], lno, me.LineNo)
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
	ItsFitnessComponent [] *KpFitnessComponent 
	Childs [] Kp
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
	st.Kparentp = len( act.ApOptimizationStrategy ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FitnessFunction has no OptimizationStrategy parent\n") ;
		return 1
	}
	st.Parent = act.ApOptimizationStrategy[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FitnessFunction under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApOptimizationStrategy[ len( act.ApOptimizationStrategy )-1 ].Childs = append(act.ApOptimizationStrategy[ len( act.ApOptimizationStrategy )-1 ].Childs, st)
	act.ApOptimizationStrategy[ len( act.ApOptimizationStrategy )-1 ].ItsFitnessFunction = append(act.ApOptimizationStrategy[ len( act.ApOptimizationStrategy )-1 ].ItsFitnessFunction, st)	// nexus.unit:648, g_structh.act:403
	name,_ := st.Names["fitness_fn"]
	s := strconv.Itoa(st.Kparentp) + "_FitnessFunction_" + name	// nexus.unit:669, g_structh.act:450
	act.index[s] = st.Me;
	st.MyName = name
	act.ApFitnessFunction = append(act.ApFitnessFunction, st)
	return 0
}

func (me KpFitnessFunction) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:648, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOptimizationStrategy[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:665, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFitnessFunction[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FitnessFunction > nexus.unit:665, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FitnessFunction > nexus.unit:665, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFitnessFunction) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "FitnessComponent" { // nexus.unit:674, g_structh.act:676
		for _, st := range me.ItsFitnessComponent {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // nexus.unit:648, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApOptimizationStrategy[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for FitnessFunction %s,%s > nexus.unit:665, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFitnessComponent struct {
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

func (me KpFitnessComponent) TypeName() string {
    return me.Comp
}
func (me KpFitnessComponent) GetLineNo() string {
	return me.LineNo
}

func loadFitnessComponent(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFitnessComponent)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFitnessComponent)
	st.LineNo = lno
	st.Comp = "FitnessComponent";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kmetricp = -1
	st.Kparentp = len( act.ApFitnessFunction ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " FitnessComponent has no FitnessFunction parent\n") ;
		return 1
	}
	st.Parent = act.ApFitnessFunction[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " FitnessComponent under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApFitnessFunction[ len( act.ApFitnessFunction )-1 ].Childs = append(act.ApFitnessFunction[ len( act.ApFitnessFunction )-1 ].Childs, st)
	act.ApFitnessFunction[ len( act.ApFitnessFunction )-1 ].ItsFitnessComponent = append(act.ApFitnessFunction[ len( act.ApFitnessFunction )-1 ].ItsFitnessComponent, st)	// nexus.unit:665, g_structh.act:403
	act.ApFitnessComponent = append(act.ApFitnessComponent, st)
	return 0
}

func (me KpFitnessComponent) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "metric" { // nexus.unit:678, g_structh.act:609
		if (me.Kmetricp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMetric[ me.Kmetricp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:665, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFitnessFunction[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:674, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFitnessComponent[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FitnessComponent > nexus.unit:674, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FitnessComponent > nexus.unit:674, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFitnessComponent) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:665, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApFitnessFunction[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "metric" {
		if me.Kmetricp >= 0 {
			st := glob.Dats.ApMetric[ me.Kmetricp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for FitnessComponent %s,%s > nexus.unit:674, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFaultModel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpFaultModel) TypeName() string {
    return me.Comp
}
func (me KpFaultModel) GetLineNo() string {
	return me.LineNo
}

func loadFaultModel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFaultModel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFaultModel)
	st.LineNo = lno
	st.Comp = "FaultModel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["fault_id"]
	act.index["FaultModel_" + name] = st.Me;
	st.MyName = name
	act.ApFaultModel = append(act.ApFaultModel, st)
	return 0
}

func (me KpFaultModel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:688, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFaultModel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FaultModel > nexus.unit:688, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FaultModel > nexus.unit:688, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFaultModel) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "Simulator_fault_models") { // nexus.unit:719, g_structh.act:583
		for _, st := range glob.Dats.ApSimulator {
			if (st.Kfault_modelsp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for FaultModel %s,%s > nexus.unit:688, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSimulator struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kproject_refp int
	Ktarget_graphp int
	Kfault_modelsp int
	ItsSimulationLevel [] *KpSimulationLevel 
	ItsAdaptiveFidelity [] *KpAdaptiveFidelity 
	ItsDeterminismConstraint [] *KpDeterminismConstraint 
	Childs [] Kp
}

func (me KpSimulator) TypeName() string {
    return me.Comp
}
func (me KpSimulator) GetLineNo() string {
	return me.LineNo
}

func loadSimulator(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSimulator)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSimulator)
	st.LineNo = lno
	st.Comp = "Simulator";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kproject_refp = -1
	st.Ktarget_graphp = -1
	st.Kfault_modelsp = -1
	name,_ := st.Names["sim_id"]
	act.index["Simulator_" + name] = st.Me;
	st.MyName = name
	act.ApSimulator = append(act.ApSimulator, st)
	return 0
}

func (me KpSimulator) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project_ref" { // nexus.unit:717, g_structh.act:609
		if (me.Kproject_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kproject_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "target_graph" { // nexus.unit:718, g_structh.act:609
		if (me.Ktarget_graphp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Ktarget_graphp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "fault_models" { // nexus.unit:719, g_structh.act:609
		if (me.Kfault_modelsp >= 0 && len(va) > 1) {
			return( glob.Dats.ApFaultModel[ me.Kfault_modelsp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "previous" { // nexus.unit:705, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSimulator[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Simulator > nexus.unit:705, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Simulator > nexus.unit:705, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSimulator) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "SimulationLevel" { // nexus.unit:721, g_structh.act:676
		for _, st := range me.ItsSimulationLevel {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "AdaptiveFidelity" { // nexus.unit:733, g_structh.act:676
		for _, st := range me.ItsAdaptiveFidelity {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "DeterminismConstraint" { // nexus.unit:741, g_structh.act:676
		for _, st := range me.ItsDeterminismConstraint {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "project_ref" {
		if me.Kproject_refp >= 0 {
			st := glob.Dats.ApProject[ me.Kproject_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "target_graph" {
		if me.Ktarget_graphp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Ktarget_graphp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "fault_models" {
		if me.Kfault_modelsp >= 0 {
			st := glob.Dats.ApFaultModel[ me.Kfault_modelsp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Simulator %s,%s > nexus.unit:705, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSimulationLevel struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	Kparentp int
	Ktarget_componentp int
}

func (me KpSimulationLevel) TypeName() string {
    return me.Comp
}
func (me KpSimulationLevel) GetLineNo() string {
	return me.LineNo
}

func loadSimulationLevel(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpSimulationLevel)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSimulationLevel)
	st.LineNo = lno
	st.Comp = "SimulationLevel";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Ktarget_componentp = -1
	st.Kparentp = len( act.ApSimulator ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " SimulationLevel has no Simulator parent\n") ;
		return 1
	}
	st.Parent = act.ApSimulator[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " SimulationLevel under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSimulator[ len( act.ApSimulator )-1 ].Childs = append(act.ApSimulator[ len( act.ApSimulator )-1 ].Childs, st)
	act.ApSimulator[ len( act.ApSimulator )-1 ].ItsSimulationLevel = append(act.ApSimulator[ len( act.ApSimulator )-1 ].ItsSimulationLevel, st)	// nexus.unit:705, g_structh.act:403
	act.ApSimulationLevel = append(act.ApSimulationLevel, st)
	return 0
}

func (me KpSimulationLevel) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "target_component" { // nexus.unit:731, g_structh.act:609
		if (me.Ktarget_componentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOperation[ me.Ktarget_componentp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // nexus.unit:705, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSimulator[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:721, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSimulationLevel[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,SimulationLevel > nexus.unit:721, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,SimulationLevel > nexus.unit:721, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpSimulationLevel) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:705, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSimulator[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "target_component" {
		if me.Ktarget_componentp >= 0 {
			st := glob.Dats.ApOperation[ me.Ktarget_componentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for SimulationLevel %s,%s > nexus.unit:721, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpAdaptiveFidelity struct {
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

func (me KpAdaptiveFidelity) TypeName() string {
    return me.Comp
}
func (me KpAdaptiveFidelity) GetLineNo() string {
	return me.LineNo
}

func loadAdaptiveFidelity(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpAdaptiveFidelity)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApAdaptiveFidelity)
	st.LineNo = lno
	st.Comp = "AdaptiveFidelity";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSimulator ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " AdaptiveFidelity has no Simulator parent\n") ;
		return 1
	}
	st.Parent = act.ApSimulator[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " AdaptiveFidelity under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSimulator[ len( act.ApSimulator )-1 ].Childs = append(act.ApSimulator[ len( act.ApSimulator )-1 ].Childs, st)
	act.ApSimulator[ len( act.ApSimulator )-1 ].ItsAdaptiveFidelity = append(act.ApSimulator[ len( act.ApSimulator )-1 ].ItsAdaptiveFidelity, st)	// nexus.unit:705, g_structh.act:403
	act.ApAdaptiveFidelity = append(act.ApAdaptiveFidelity, st)
	return 0
}

func (me KpAdaptiveFidelity) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:705, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSimulator[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:733, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApAdaptiveFidelity[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,AdaptiveFidelity > nexus.unit:733, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,AdaptiveFidelity > nexus.unit:733, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpAdaptiveFidelity) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:705, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSimulator[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for AdaptiveFidelity %s,%s > nexus.unit:733, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDeterminismConstraint struct {
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

func (me KpDeterminismConstraint) TypeName() string {
    return me.Comp
}
func (me KpDeterminismConstraint) GetLineNo() string {
	return me.LineNo
}

func loadDeterminismConstraint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDeterminismConstraint)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDeterminismConstraint)
	st.LineNo = lno
	st.Comp = "DeterminismConstraint";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApSimulator ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " DeterminismConstraint has no Simulator parent\n") ;
		return 1
	}
	st.Parent = act.ApSimulator[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " DeterminismConstraint under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApSimulator[ len( act.ApSimulator )-1 ].Childs = append(act.ApSimulator[ len( act.ApSimulator )-1 ].Childs, st)
	act.ApSimulator[ len( act.ApSimulator )-1 ].ItsDeterminismConstraint = append(act.ApSimulator[ len( act.ApSimulator )-1 ].ItsDeterminismConstraint, st)	// nexus.unit:705, g_structh.act:403
	act.ApDeterminismConstraint = append(act.ApDeterminismConstraint, st)
	return 0
}

func (me KpDeterminismConstraint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:705, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApSimulator[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:741, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDeterminismConstraint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DeterminismConstraint > nexus.unit:741, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DeterminismConstraint > nexus.unit:741, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDeterminismConstraint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:705, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApSimulator[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for DeterminismConstraint %s,%s > nexus.unit:741, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
}

func (me KpConstraint) TypeName() string {
    return me.Comp
}
func (me KpConstraint) GetLineNo() string {
	return me.LineNo
}

func loadConstraint(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["constraint_id"]
	act.index["Constraint_" + name] = st.Me;
	st.MyName = name
	act.ApConstraint = append(act.ApConstraint, st)
	return 0
}

func (me KpConstraint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:754, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApConstraint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Constraint > nexus.unit:754, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Constraint > nexus.unit:754, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpConstraint) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for Constraint %s,%s > nexus.unit:754, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Names map[string]string
}

func (me KpMetric) TypeName() string {
    return me.Comp
}
func (me KpMetric) GetLineNo() string {
	return me.LineNo
}

func loadMetric(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
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
	name,_ := st.Names["metric"]
	act.index["Metric_" + name] = st.Me;
	st.MyName = name
	act.ApMetric = append(act.ApMetric, st)
	return 0
}

func (me KpMetric) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "ProfilingHook_metrics_ref" && len(va) > 1) { // nexus.unit:266, g_structh.act:698
		for _, st := range glob.Dats.ApProfilingHook {
			if (st.Kmetrics_refp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "EvolvableGraph_fitness_metric" && len(va) > 1) { // nexus.unit:395, g_structh.act:698
		for _, st := range glob.Dats.ApEvolvableGraph {
			if (st.Kfitness_metricp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "FitnessComponent_metric" && len(va) > 1) { // nexus.unit:678, g_structh.act:698
		for _, st := range glob.Dats.ApFitnessComponent {
			if (st.Kmetricp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:771, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMetric[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Metric > nexus.unit:771, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Metric > nexus.unit:771, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpMetric) DoIts(glob *GlobT, va []string, lno string) int {
	if (va[0] == "ProfilingHook_metrics_ref") { // nexus.unit:266, g_structh.act:583
		for _, st := range glob.Dats.ApProfilingHook {
			if (st.Kmetrics_refp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "EvolvableGraph_fitness_metric") { // nexus.unit:395, g_structh.act:583
		for _, st := range glob.Dats.ApEvolvableGraph {
			if (st.Kfitness_metricp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "FitnessComponent_metric") { // nexus.unit:678, g_structh.act:583
		for _, st := range glob.Dats.ApFitnessComponent {
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
	        fmt.Printf("?No its %s for Metric %s,%s > nexus.unit:771, g_structh.act:209?", va[0], lno, me.LineNo)
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
	Kproject_refp int
	Kproject_sourcep int
	Kgraph_snapshotp int
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
	st.Kproject_refp = -1
	st.Kproject_sourcep = -1
	st.Kgraph_snapshotp = -1
	name,_ := st.Names["checkpoint_id"]
	act.index["Checkpoint_" + name] = st.Me;
	st.MyName = name
	act.ApCheckpoint = append(act.ApCheckpoint, st)
	return 0
}

func (me KpCheckpoint) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "project_ref" { // nexus.unit:793, g_structh.act:609
		if (me.Kproject_refp >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kproject_refp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "project_source" { // nexus.unit:798, g_structh.act:609
		if (me.Kproject_sourcep >= 0 && len(va) > 1) {
			return( glob.Dats.ApProject[ me.Kproject_sourcep ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "graph_snapshot" { // nexus.unit:799, g_structh.act:609
		if (me.Kgraph_snapshotp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComputeGraph[ me.Kgraph_snapshotp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "RedundancyStrategy_checkpoints" && len(va) > 1) { // nexus.unit:255, g_structh.act:698
		for _, st := range glob.Dats.ApRedundancyStrategy {
			if (st.Kcheckpointsp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "OptimizationRun_best_checkpoint" && len(va) > 1) { // nexus.unit:422, g_structh.act:698
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kbest_checkpointp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // nexus.unit:788, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCheckpoint[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Checkpoint > nexus.unit:788, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Checkpoint > nexus.unit:788, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCheckpoint) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "project_ref" {
		if me.Kproject_refp >= 0 {
			st := glob.Dats.ApProject[ me.Kproject_refp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "project_source" {
		if me.Kproject_sourcep >= 0 {
			st := glob.Dats.ApProject[ me.Kproject_sourcep ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "graph_snapshot" {
		if me.Kgraph_snapshotp >= 0 {
			st := glob.Dats.ApComputeGraph[ me.Kgraph_snapshotp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "RedundancyStrategy_checkpoints") { // nexus.unit:255, g_structh.act:583
		for _, st := range glob.Dats.ApRedundancyStrategy {
			if (st.Kcheckpointsp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	if (va[0] == "OptimizationRun_best_checkpoint") { // nexus.unit:422, g_structh.act:583
		for _, st := range glob.Dats.ApOptimizationRun {
			if (st.Kbest_checkpointp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
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
	        fmt.Printf("?No its %s for Checkpoint %s,%s > nexus.unit:788, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpParadigmRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpParadigmRule) TypeName() string {
    return me.Comp
}
func (me KpParadigmRule) GetLineNo() string {
	return me.LineNo
}

func loadParadigmRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpParadigmRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApParadigmRule)
	st.LineNo = lno
	st.Comp = "ParadigmRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["paradigm"]
	act.index["ParadigmRule_" + name] = st.Me;
	st.MyName = name
	act.ApParadigmRule = append(act.ApParadigmRule, st)
	return 0
}

func (me KpParadigmRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:805, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApParadigmRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,ParadigmRule > nexus.unit:805, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,ParadigmRule > nexus.unit:805, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpParadigmRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for ParadigmRule %s,%s > nexus.unit:805, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpNeuronModelRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpNeuronModelRule) TypeName() string {
    return me.Comp
}
func (me KpNeuronModelRule) GetLineNo() string {
	return me.LineNo
}

func loadNeuronModelRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpNeuronModelRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApNeuronModelRule)
	st.LineNo = lno
	st.Comp = "NeuronModelRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["model"]
	act.index["NeuronModelRule_" + name] = st.Me;
	st.MyName = name
	act.ApNeuronModelRule = append(act.ApNeuronModelRule, st)
	return 0
}

func (me KpNeuronModelRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:812, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApNeuronModelRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,NeuronModelRule > nexus.unit:812, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,NeuronModelRule > nexus.unit:812, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpNeuronModelRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for NeuronModelRule %s,%s > nexus.unit:812, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpQuantumGateRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpQuantumGateRule) TypeName() string {
    return me.Comp
}
func (me KpQuantumGateRule) GetLineNo() string {
	return me.LineNo
}

func loadQuantumGateRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpQuantumGateRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApQuantumGateRule)
	st.LineNo = lno
	st.Comp = "QuantumGateRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["gate"]
	act.index["QuantumGateRule_" + name] = st.Me;
	st.MyName = name
	act.ApQuantumGateRule = append(act.ApQuantumGateRule, st)
	return 0
}

func (me KpQuantumGateRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:821, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApQuantumGateRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,QuantumGateRule > nexus.unit:821, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,QuantumGateRule > nexus.unit:821, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpQuantumGateRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for QuantumGateRule %s,%s > nexus.unit:821, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpDataflowRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpDataflowRule) TypeName() string {
    return me.Comp
}
func (me KpDataflowRule) GetLineNo() string {
	return me.LineNo
}

func loadDataflowRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpDataflowRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApDataflowRule)
	st.LineNo = lno
	st.Comp = "DataflowRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["dataflow"]
	act.index["DataflowRule_" + name] = st.Me;
	st.MyName = name
	act.ApDataflowRule = append(act.ApDataflowRule, st)
	return 0
}

func (me KpDataflowRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:830, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApDataflowRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,DataflowRule > nexus.unit:830, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,DataflowRule > nexus.unit:830, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpDataflowRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for DataflowRule %s,%s > nexus.unit:830, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpFusionStrategyRule struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
}

func (me KpFusionStrategyRule) TypeName() string {
    return me.Comp
}
func (me KpFusionStrategyRule) GetLineNo() string {
	return me.LineNo
}

func loadFusionStrategyRule(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpFusionStrategyRule)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApFusionStrategyRule)
	st.LineNo = lno
	st.Comp = "FusionStrategyRule";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["strategy"]
	act.index["FusionStrategyRule_" + name] = st.Me;
	st.MyName = name
	act.ApFusionStrategyRule = append(act.ApFusionStrategyRule, st)
	return 0
}

func (me KpFusionStrategyRule) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:838, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApFusionStrategyRule[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,FusionStrategyRule > nexus.unit:838, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,FusionStrategyRule > nexus.unit:838, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpFusionStrategyRule) DoIts(glob *GlobT, va []string, lno string) int {
	        fmt.Printf("?No its %s for FusionStrategyRule %s,%s > nexus.unit:838, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpExtension struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]string
	ItsCustomParadigm [] *KpCustomParadigm 
	Childs [] Kp
}

func (me KpExtension) TypeName() string {
    return me.Comp
}
func (me KpExtension) GetLineNo() string {
	return me.LineNo
}

func loadExtension(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpExtension)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApExtension)
	st.LineNo = lno
	st.Comp = "Extension";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["ext_name"]
	act.index["Extension_" + name] = st.Me;
	st.MyName = name
	act.ApExtension = append(act.ApExtension, st)
	return 0
}

func (me KpExtension) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "previous" { // nexus.unit:850, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApExtension[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Extension > nexus.unit:850, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,Extension > nexus.unit:850, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpExtension) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "CustomParadigm" { // nexus.unit:861, g_structh.act:676
		for _, st := range me.ItsCustomParadigm {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	        fmt.Printf("?No its %s for Extension %s,%s > nexus.unit:850, g_structh.act:209?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCustomParadigm struct {
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

func (me KpCustomParadigm) TypeName() string {
    return me.Comp
}
func (me KpCustomParadigm) GetLineNo() string {
	return me.LineNo
}

func loadCustomParadigm(act *ActT, ln string, pos int, lno string, flag []string, names map[string]string) int {
	st := new(KpCustomParadigm)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCustomParadigm)
	st.LineNo = lno
	st.Comp = "CustomParadigm";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApExtension ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " CustomParadigm has no Extension parent\n") ;
		return 1
	}
	st.Parent = act.ApExtension[st.Kparentp].MyName
	par,ok := st.Names["parent"]
	if ok && par != st.Parent {
		print(lno + " CustomParadigm under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApExtension[ len( act.ApExtension )-1 ].Childs = append(act.ApExtension[ len( act.ApExtension )-1 ].Childs, st)
	act.ApExtension[ len( act.ApExtension )-1 ].ItsCustomParadigm = append(act.ApExtension[ len( act.ApExtension )-1 ].ItsCustomParadigm, st)	// nexus.unit:850, g_structh.act:403
	act.ApCustomParadigm = append(act.ApCustomParadigm, st)
	return 0
}

func (me KpCustomParadigm) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // nexus.unit:850, g_structh.act:572
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApExtension[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // nexus.unit:861, g_structh.act:176
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCustomParadigm[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,CustomParadigm > nexus.unit:861, g_structh.act:183?", va[0], lno, me.LineNo)
		return false, msg
	}
	r,ok := me.Names[va[0]]
	if !ok { r = fmt.Sprintf("?%s?:%s,%s,CustomParadigm > nexus.unit:861, g_structh.act:187?", va[0], lno, me.LineNo) }
	return ok,r
}

func (me KpCustomParadigm) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // nexus.unit:850, g_structh.act:557
		if me.Kparentp >= 0 {
			st := glob.Dats.ApExtension[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for CustomParadigm %s,%s > nexus.unit:861, g_structh.act:209?", va[0], lno, me.LineNo)
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

