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

type KpComp struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	ItsElement [] *KpElement 
	Childs [] Kp
}

func (me KpComp) TypeName() string {
    return me.Comp
}
func (me KpComp) GetLineNo() string {
	return me.LineNo
}

func loadComp(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpComp)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApComp)
	st.LineNo = lno
	st.Comp = "Comp";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = -1
	name,_ := st.Names["name"].(string)
	st.Names["_key"] = "name"
	act.index["Comp_" + name] = st.Me;
	st.MyName = name
	act.ApComp = append(act.ApComp, st)
	return 0
}

func (me KpComp) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "parent" { // one.unit:8, go-struct-rio.act:621
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComp[ me.Kparentp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Comp_parent" && len(va) > 1) { // one.unit:8, go-struct-rio.act:706
		for _, st := range glob.Dats.ApComp {
			if (st.Kparentp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Element_comp" && len(va) > 1) { // one.unit:36, go-struct-rio.act:706
		for _, st := range glob.Dats.ApElement {
			if (st.Kcompp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // one.unit:2, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApComp[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Comp > one.unit:2, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Comp > one.unit:2, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpComp) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Element" { // one.unit:18, go-struct-rio.act:685
		for _, st := range me.ItsElement {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" {
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Comp_parent") { // one.unit:8, go-struct-rio.act:595
		for _, st := range glob.Dats.ApComp {
			if (st.Kparentp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Element_comp") { // one.unit:36, go-struct-rio.act:595
		for _, st := range glob.Dats.ApElement {
			if (st.Kcompp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Comp %s,%s > one.unit:2, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpElement struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kcompp int
	ItsOpt [] *KpOpt 
	Childs [] Kp
}

func (me KpElement) TypeName() string {
    return me.Comp
}
func (me KpElement) GetLineNo() string {
	return me.LineNo
}

func loadElement(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpElement)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApElement)
	st.LineNo = lno
	st.Comp = "Element";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcompp = -1
	st.Kparentp = len( act.ApComp ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Element has no Comp parent\n") ;
		return 1
	}
	st.Parent = act.ApComp[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Element under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApComp[ len( act.ApComp )-1 ].Childs = append(act.ApComp[ len( act.ApComp )-1 ].Childs, st)
	act.ApComp[ len( act.ApComp )-1 ].ItsElement = append(act.ApComp[ len( act.ApComp )-1 ].ItsElement, st)	// one.unit:2, go-struct-rio.act:416
	name,_ := st.Names["name"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Element_" + name	// one.unit:20, go-struct-rio.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApElement = append(act.ApElement, st)
	return 0
}

func (me KpElement) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "comp" { // one.unit:36, go-struct-rio.act:621
		if (me.Kcompp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComp[ me.Kcompp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // one.unit:2, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApComp[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // one.unit:18, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApElement[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Element > one.unit:18, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Element > one.unit:18, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpElement) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Opt" { // one.unit:50, go-struct-rio.act:685
		for _, st := range me.ItsOpt {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // one.unit:2, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApComp[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "comp" {
		if me.Kcompp >= 0 {
			st := glob.Dats.ApComp[ me.Kcompp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Element %s,%s > one.unit:18, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOpt struct {
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

func (me KpOpt) TypeName() string {
    return me.Comp
}
func (me KpOpt) GetLineNo() string {
	return me.LineNo
}

func loadOpt(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpOpt)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOpt)
	st.LineNo = lno
	st.Comp = "Opt";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApElement ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Opt has no Element parent\n") ;
		return 1
	}
	st.Parent = act.ApElement[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Opt under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApElement[ len( act.ApElement )-1 ].Childs = append(act.ApElement[ len( act.ApElement )-1 ].Childs, st)
	act.ApElement[ len( act.ApElement )-1 ].ItsOpt = append(act.ApElement[ len( act.ApElement )-1 ].ItsOpt, st)	// one.unit:18, go-struct-rio.act:416
	name,_ := st.Names["name"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Opt_" + name	// one.unit:56, go-struct-rio.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApOpt = append(act.ApOpt, st)
	return 0
}

func (me KpOpt) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // one.unit:18, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApElement[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // one.unit:50, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOpt[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Opt > one.unit:50, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Opt > one.unit:50, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpOpt) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // one.unit:18, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApElement[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Opt %s,%s > one.unit:50, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpObjective struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kcanonical_formp int
	ItsMemory [] *KpMemory 
	ItsThought [] *KpThought 
	Childs [] Kp
}

func (me KpObjective) TypeName() string {
    return me.Comp
}
func (me KpObjective) GetLineNo() string {
	return me.LineNo
}

func loadObjective(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpObjective)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApObjective)
	st.LineNo = lno
	st.Comp = "Objective";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcanonical_formp = -1
	name,_ := st.Names["objective_id"].(string)
	st.Names["_key"] = "objective_id"
	act.index["Objective_" + name] = st.Me;
	st.MyName = name
	act.ApObjective = append(act.ApObjective, st)
	return 0
}

func (me KpObjective) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "canonical_form" { // one.unit:73, go-struct-rio.act:621
		if (me.Kcanonical_formp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kcanonical_formp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Memo_objective_id" && len(va) > 1) { // one.unit:117, go-struct-rio.act:706
		for _, st := range glob.Dats.ApMemo {
			if (st.Kobjective_idp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // one.unit:70, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApObjective[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Objective > one.unit:70, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Objective > one.unit:70, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpObjective) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Memory" { // one.unit:81, go-struct-rio.act:685
		for _, st := range me.ItsMemory {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Thought" { // one.unit:96, go-struct-rio.act:685
		for _, st := range me.ItsThought {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "canonical_form" {
		if me.Kcanonical_formp >= 0 {
			st := glob.Dats.ApCanon[ me.Kcanonical_formp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Memo_objective_id") { // one.unit:117, go-struct-rio.act:595
		for _, st := range glob.Dats.ApMemo {
			if (st.Kobjective_idp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Objective %s,%s > one.unit:70, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMemory struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kinvalidated_byp int
}

func (me KpMemory) TypeName() string {
    return me.Comp
}
func (me KpMemory) GetLineNo() string {
	return me.LineNo
}

func loadMemory(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpMemory)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMemory)
	st.LineNo = lno
	st.Comp = "Memory";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kinvalidated_byp = -1
	st.Kparentp = len( act.ApObjective ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Memory has no Objective parent\n") ;
		return 1
	}
	st.Parent = act.ApObjective[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Memory under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApObjective[ len( act.ApObjective )-1 ].Childs = append(act.ApObjective[ len( act.ApObjective )-1 ].Childs, st)
	act.ApObjective[ len( act.ApObjective )-1 ].ItsMemory = append(act.ApObjective[ len( act.ApObjective )-1 ].ItsMemory, st)	// one.unit:70, go-struct-rio.act:416
	name,_ := st.Names["memory_id"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Memory_" + name	// one.unit:82, go-struct-rio.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApMemory = append(act.ApMemory, st)
	return 0
}

func (me KpMemory) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "invalidated_by" { // one.unit:87, go-struct-rio.act:621
		if (me.Kinvalidated_byp >= 0 && len(va) > 1) {
			return( glob.Dats.ApThought[ me.Kinvalidated_byp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // one.unit:70, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApObjective[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // one.unit:81, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMemory[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Memory > one.unit:81, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Memory > one.unit:81, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpMemory) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // one.unit:70, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApObjective[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "invalidated_by" {
		if me.Kinvalidated_byp >= 0 {
			st := glob.Dats.ApThought[ me.Kinvalidated_byp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Memo_memory_id") { // one.unit:118, go-struct-rio.act:595
		for _, st := range glob.Dats.ApMemo {
			if (st.Kmemory_idp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Memory %s,%s > one.unit:81, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpThought struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kcanonical_formp int
	ItsMemo [] *KpMemo 
	Childs [] Kp
}

func (me KpThought) TypeName() string {
    return me.Comp
}
func (me KpThought) GetLineNo() string {
	return me.LineNo
}

func loadThought(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpThought)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApThought)
	st.LineNo = lno
	st.Comp = "Thought";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcanonical_formp = -1
	st.Kparentp = len( act.ApObjective ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Thought has no Objective parent\n") ;
		return 1
	}
	st.Parent = act.ApObjective[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Thought under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApObjective[ len( act.ApObjective )-1 ].Childs = append(act.ApObjective[ len( act.ApObjective )-1 ].Childs, st)
	act.ApObjective[ len( act.ApObjective )-1 ].ItsThought = append(act.ApObjective[ len( act.ApObjective )-1 ].ItsThought, st)	// one.unit:70, go-struct-rio.act:416
	name,_ := st.Names["thought_id"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Thought_" + name	// one.unit:97, go-struct-rio.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApThought = append(act.ApThought, st)
	return 0
}

func (me KpThought) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "canonical_form" { // one.unit:112, go-struct-rio.act:621
		if (me.Kcanonical_formp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kcanonical_formp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // one.unit:70, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApObjective[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if (va[0] == "Memory_invalidated_by" && len(va) > 1) { // one.unit:87, go-struct-rio.act:706
		for _, st := range glob.Dats.ApMemory {
			if (st.Kinvalidated_byp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // one.unit:96, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApThought[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Thought > one.unit:96, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Thought > one.unit:96, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpThought) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Memo" { // one.unit:114, go-struct-rio.act:685
		for _, st := range me.ItsMemo {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "parent" { // one.unit:70, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApObjective[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "canonical_form" {
		if me.Kcanonical_formp >= 0 {
			st := glob.Dats.ApCanon[ me.Kcanonical_formp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Memory_invalidated_by") { // one.unit:87, go-struct-rio.act:595
		for _, st := range glob.Dats.ApMemory {
			if (st.Kinvalidated_byp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
		if va[0] == "tree_linear" && len(va) > 1 && me.Kparentp >= 0 { // one.unit:98, go-struct-rio.act:233
			pos := 0
			v, ok := me.Names["tree_linear"].(string)
			if ok {
				pos, _ = strconv.Atoi(v)
			}
			if va[1] == "down" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				for _, st := range pst.ItsThought {
					if st.Me == me.Me {
						isin = true
						continue
					}
					if !isin {
						continue
					}
					pos2 := 0
					v2, ok2 := st.Names["tree_linear"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == (pos - 1) {
						break
					}
					if pos2 == pos {
						if len(va) > 2 {
							return st.DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, st)
					}
				}
				return 0
			}
			if va[1] == "up" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				prev := 0
				for _, st := range pst.ItsThought {
					pos2 := 0
					v2, ok2 := st.Names["tree_linear"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == pos && st.Me != me.Me {
						prev = st.Me
						isin = true
						continue
					}
					if pos2 == (pos - 1) {
						isin = false
					}
					if st.Me == me.Me && isin {
						if len(va) > 2 {
							return glob.Dats.ApThought[prev].DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, glob.Dats.ApThought[prev])
					}
				}
				return 0
			}
			if va[1] == "left" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				prev := 0
				for _, st := range pst.ItsThought {
					pos2 := 0
					v2, ok2 := st.Names["tree_linear"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == (pos - 1) {
						prev = st.Me
						isin = true
						continue
					}
					if st.Me == me.Me && isin {
						if len(va) > 2 {
							return glob.Dats.ApThought[prev].DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, glob.Dats.ApThought[prev])
					}
				}
				return 0
			}
			if va[1] == "right" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				for _, st := range pst.ItsThought {
					if st.Me == me.Me {
						isin = true
						continue
					}
					if !isin {
						continue
					}
					pos2 := 0
					v2, ok2 := st.Names["tree_linear"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 <= pos {
						break
					}
					if pos2 == (pos + 1) {
						if len(va) > 2 {
							ret := st.DoIts(glob, va[2:], lno)
							if ret != 0 {
								return ret
							}
							continue
						}
						ret := GoAct(glob, st)
						if ret != 0 {
							return ret
						}
					}
				}
				return 0
			}
		}
		if va[0] == "tree_parallel" && len(va) > 1 && me.Kparentp >= 0 { // one.unit:99, go-struct-rio.act:233
			pos := 0
			v, ok := me.Names["tree_parallel"].(string)
			if ok {
				pos, _ = strconv.Atoi(v)
			}
			if va[1] == "down" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				for _, st := range pst.ItsThought {
					if st.Me == me.Me {
						isin = true
						continue
					}
					if !isin {
						continue
					}
					pos2 := 0
					v2, ok2 := st.Names["tree_parallel"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == (pos - 1) {
						break
					}
					if pos2 == pos {
						if len(va) > 2 {
							return st.DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, st)
					}
				}
				return 0
			}
			if va[1] == "up" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				prev := 0
				for _, st := range pst.ItsThought {
					pos2 := 0
					v2, ok2 := st.Names["tree_parallel"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == pos && st.Me != me.Me {
						prev = st.Me
						isin = true
						continue
					}
					if pos2 == (pos - 1) {
						isin = false
					}
					if st.Me == me.Me && isin {
						if len(va) > 2 {
							return glob.Dats.ApThought[prev].DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, glob.Dats.ApThought[prev])
					}
				}
				return 0
			}
			if va[1] == "left" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				prev := 0
				for _, st := range pst.ItsThought {
					pos2 := 0
					v2, ok2 := st.Names["tree_parallel"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == (pos - 1) {
						prev = st.Me
						isin = true
						continue
					}
					if st.Me == me.Me && isin {
						if len(va) > 2 {
							return glob.Dats.ApThought[prev].DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, glob.Dats.ApThought[prev])
					}
				}
				return 0
			}
			if va[1] == "right" && pos > 0 {
				pst := glob.Dats.ApObjective[me.Kparentp]
				isin := false
				for _, st := range pst.ItsThought {
					if st.Me == me.Me {
						isin = true
						continue
					}
					if !isin {
						continue
					}
					pos2 := 0
					v2, ok2 := st.Names["tree_parallel"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 <= pos {
						break
					}
					if pos2 == (pos + 1) {
						if len(va) > 2 {
							ret := st.DoIts(glob, va[2:], lno)
							if ret != 0 {
								return ret
							}
							continue
						}
						ret := GoAct(glob, st)
						if ret != 0 {
							return ret
						}
					}
				}
				return 0
			}
		}
	        fmt.Printf("?No its %s for Thought %s,%s > one.unit:96, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpMemo struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kcanonical_formp int
	Kobjective_idp int
	Kmemory_idp int
	Kcomputationp int
}

func (me KpMemo) TypeName() string {
    return me.Comp
}
func (me KpMemo) GetLineNo() string {
	return me.LineNo
}

func loadMemo(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpMemo)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApMemo)
	st.LineNo = lno
	st.Comp = "Memo";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcanonical_formp = -1
	st.Kobjective_idp = -1
	st.Kmemory_idp = -1
	st.Kcomputationp = -1
	st.Kparentp = len( act.ApThought ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Memo has no Thought parent\n") ;
		return 1
	}
	st.Parent = act.ApThought[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Memo under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApThought[ len( act.ApThought )-1 ].Childs = append(act.ApThought[ len( act.ApThought )-1 ].Childs, st)
	act.ApThought[ len( act.ApThought )-1 ].ItsMemo = append(act.ApThought[ len( act.ApThought )-1 ].ItsMemo, st)	// one.unit:96, go-struct-rio.act:416
	name,_ := st.Names["memo_id"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Memo_" + name	// one.unit:115, go-struct-rio.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApMemo = append(act.ApMemo, st)
	return 0
}

func (me KpMemo) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "canonical_form" { // one.unit:116, go-struct-rio.act:621
		if (me.Kcanonical_formp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kcanonical_formp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "objective_id" { // one.unit:117, go-struct-rio.act:621
		if (me.Kobjective_idp >= 0 && len(va) > 1) {
			return( glob.Dats.ApObjective[ me.Kobjective_idp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "memory_id" { // one.unit:118, go-struct-rio.act:621
		if (me.Kmemory_idp >= 0 && len(va) > 1) {
			return( glob.Dats.ApMemory[ me.Kmemory_idp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "computation" { // one.unit:119, go-struct-rio.act:621
		if (me.Kcomputationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApOps[ me.Kcomputationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // one.unit:96, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApThought[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // one.unit:114, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApMemo[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Memo > one.unit:114, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Memo > one.unit:114, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpMemo) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // one.unit:96, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApThought[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "canonical_form" {
		if me.Kcanonical_formp >= 0 {
			st := glob.Dats.ApCanon[ me.Kcanonical_formp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "objective_id" {
		if me.Kobjective_idp >= 0 {
			st := glob.Dats.ApObjective[ me.Kobjective_idp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "memory_id" {
		if me.Kmemory_idp >= 0 {
			st := glob.Dats.ApMemory[ me.Kmemory_idp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "computation" {
		if me.Kcomputationp >= 0 {
			st := glob.Dats.ApOps[ me.Kcomputationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Memo %s,%s > one.unit:114, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpOps struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kcanonical_formp int
}

func (me KpOps) TypeName() string {
    return me.Comp
}
func (me KpOps) GetLineNo() string {
	return me.LineNo
}

func loadOps(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpOps)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApOps)
	st.LineNo = lno
	st.Comp = "Ops";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kcanonical_formp = -1
	name,_ := st.Names["ops_id"].(string)
	st.Names["_key"] = "ops_id"
	act.index["Ops_" + name] = st.Me;
	st.MyName = name
	act.ApOps = append(act.ApOps, st)
	return 0
}

func (me KpOps) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "canonical_form" { // one.unit:133, go-struct-rio.act:621
		if (me.Kcanonical_formp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kcanonical_formp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "Memo_computation" && len(va) > 1) { // one.unit:119, go-struct-rio.act:706
		for _, st := range glob.Dats.ApMemo {
			if (st.Kcomputationp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // one.unit:121, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApOps[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Ops > one.unit:121, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Ops > one.unit:121, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpOps) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "canonical_form" {
		if me.Kcanonical_formp >= 0 {
			st := glob.Dats.ApCanon[ me.Kcanonical_formp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if (va[0] == "Memo_computation") { // one.unit:119, go-struct-rio.act:595
		for _, st := range glob.Dats.ApMemo {
			if (st.Kcomputationp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Ops %s,%s > one.unit:121, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpCanon struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	ItsLink [] *KpLink 
	ItsSection [] *KpSection 
	Childs [] Kp
}

func (me KpCanon) TypeName() string {
    return me.Comp
}
func (me KpCanon) GetLineNo() string {
	return me.LineNo
}

func loadCanon(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpCanon)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApCanon)
	st.LineNo = lno
	st.Comp = "Canon";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	name,_ := st.Names["name"].(string)
	st.Names["_key"] = "name"
	act.index["Canon_" + name] = st.Me;
	st.MyName = name
	act.ApCanon = append(act.ApCanon, st)
	return 0
}

func (me KpCanon) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "Objective_canonical_form" && len(va) > 1) { // one.unit:73, go-struct-rio.act:706
		for _, st := range glob.Dats.ApObjective {
			if (st.Kcanonical_formp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Thought_canonical_form" && len(va) > 1) { // one.unit:112, go-struct-rio.act:706
		for _, st := range glob.Dats.ApThought {
			if (st.Kcanonical_formp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Memo_canonical_form" && len(va) > 1) { // one.unit:116, go-struct-rio.act:706
		for _, st := range glob.Dats.ApMemo {
			if (st.Kcanonical_formp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Ops_canonical_form" && len(va) > 1) { // one.unit:133, go-struct-rio.act:706
		for _, st := range glob.Dats.ApOps {
			if (st.Kcanonical_formp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Link_concept" && len(va) > 1) { // one.unit:158, go-struct-rio.act:706
		for _, st := range glob.Dats.ApLink {
			if (st.Kconceptp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if (va[0] == "Link_relation" && len(va) > 1) { // one.unit:159, go-struct-rio.act:706
		for _, st := range glob.Dats.ApLink {
			if (st.Krelationp == me.Me) {
				return (st.GetVar(glob, va[1:], lno) )
			}
		}
	}
	if va[0] == "previous" { // one.unit:141, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Canon > one.unit:141, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Canon > one.unit:141, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpCanon) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "Link" { // one.unit:153, go-struct-rio.act:685
		for _, st := range me.ItsLink {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if va[0] == "Section" { // one.unit:161, go-struct-rio.act:685
		for _, st := range me.ItsSection {
			if len(va) > 1 {
				ret := st.DoIts(glob, va[1:], lno)
				if (ret != 0) {
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
	if (va[0] == "Objective_canonical_form") { // one.unit:73, go-struct-rio.act:595
		for _, st := range glob.Dats.ApObjective {
			if (st.Kcanonical_formp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Thought_canonical_form") { // one.unit:112, go-struct-rio.act:595
		for _, st := range glob.Dats.ApThought {
			if (st.Kcanonical_formp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Memo_canonical_form") { // one.unit:116, go-struct-rio.act:595
		for _, st := range glob.Dats.ApMemo {
			if (st.Kcanonical_formp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Ops_canonical_form") { // one.unit:133, go-struct-rio.act:595
		for _, st := range glob.Dats.ApOps {
			if (st.Kcanonical_formp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Link_concept") { // one.unit:158, go-struct-rio.act:595
		for _, st := range glob.Dats.ApLink {
			if (st.Kconceptp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	if (va[0] == "Link_relation") { // one.unit:159, go-struct-rio.act:595
		for _, st := range glob.Dats.ApLink {
			if (st.Krelationp == me.Me) {
				if len(va) > 1 {
					ret := st.DoIts(glob, va[1:], lno)
					if (ret != 0) {
						return(ret)
					}
					continue
				}
				ret := GoAct(glob, st)
				if (ret != 0) {
					return(ret)
				}
			}
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Canon %s,%s > one.unit:141, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpLink struct {
	Kp
	Me int
	MyName string
	Parent string
	LineNo string
	Comp string
	Flags [] string
	Names map[string]any
	Kparentp int
	Kconceptp int
	Krelationp int
}

func (me KpLink) TypeName() string {
    return me.Comp
}
func (me KpLink) GetLineNo() string {
	return me.LineNo
}

func loadLink(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpLink)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApLink)
	st.LineNo = lno
	st.Comp = "Link";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kconceptp = -1
	st.Krelationp = -1
	st.Kparentp = len( act.ApCanon ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Link has no Canon parent\n") ;
		return 1
	}
	st.Parent = act.ApCanon[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Link under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApCanon[ len( act.ApCanon )-1 ].Childs = append(act.ApCanon[ len( act.ApCanon )-1 ].Childs, st)
	act.ApCanon[ len( act.ApCanon )-1 ].ItsLink = append(act.ApCanon[ len( act.ApCanon )-1 ].ItsLink, st)	// one.unit:141, go-struct-rio.act:416
	act.ApLink = append(act.ApLink, st)
	return 0
}

func (me KpLink) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if va[0] == "concept" { // one.unit:158, go-struct-rio.act:621
		if (me.Kconceptp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kconceptp ].GetVar(glob, va[1:], lno) )
		}
	}
	if va[0] == "relation" { // one.unit:159, go-struct-rio.act:621
		if (me.Krelationp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Krelationp ].GetVar(glob, va[1:], lno) )
		}
	}
	if (va[0] == "parent") { // one.unit:141, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // one.unit:153, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApLink[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Link > one.unit:153, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Link > one.unit:153, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpLink) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // one.unit:141, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApCanon[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "concept" {
		if me.Kconceptp >= 0 {
			st := glob.Dats.ApCanon[ me.Kconceptp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	if va[0] == "relation" {
		if me.Krelationp >= 0 {
			st := glob.Dats.ApCanon[ me.Krelationp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
	        fmt.Printf("?No its %s for Link %s,%s > one.unit:153, go-struct-rio.act:222?", va[0], lno, me.LineNo)
		glob.RunErrs += 1
	return(0)
}

type KpSection struct {
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

func (me KpSection) TypeName() string {
    return me.Comp
}
func (me KpSection) GetLineNo() string {
	return me.LineNo
}

func loadSection(act *ActT, ln string, pos int, lno string, flag []string, names map[string]any) int {
	st := new(KpSection)
	st.Names = names
      st.MyName = ""
      st.Parent = ""
	st.Me = len(act.ApSection)
	st.LineNo = lno
	st.Comp = "Section";
	st.Flags = flag;
	st.Names["kComp"] = st.Comp
	st.Names["kMe"] = strconv.Itoa(st.Me)
	st.Names["_lno"] = lno
	st.Kparentp = len( act.ApCanon ) - 1;
	st.Names["kParentp"] = strconv.Itoa(st.Kparentp)
	if (st.Kparentp < 0 ) { 
		print(lno + " Section has no Canon parent\n") ;
		return 1
	}
	st.Parent = act.ApCanon[st.Kparentp].MyName
	par,ok := st.Names["parent"].(string)
	if ok && par != st.Parent {
		print(lno + " Section under wrong parent " + st.Parent + ", " +  par + "\n") ;
	}
	act.ApCanon[ len( act.ApCanon )-1 ].Childs = append(act.ApCanon[ len( act.ApCanon )-1 ].Childs, st)
	act.ApCanon[ len( act.ApCanon )-1 ].ItsSection = append(act.ApCanon[ len( act.ApCanon )-1 ].ItsSection, st)	// one.unit:141, go-struct-rio.act:416
	name,_ := st.Names["name"].(string)
	s := strconv.Itoa(st.Kparentp) + "_Section_" + name	// one.unit:165, go-struct-rio.act:464
	act.index[s] = st.Me;
	st.MyName = name
	act.ApSection = append(act.ApSection, st)
	return 0
}

func (me KpSection) GetVar(glob *GlobT, va []string, lno string) (bool, string) {
	if (va[0] == "parent") { // one.unit:141, go-struct-rio.act:585
		if (me.Kparentp >= 0 && len(va) > 1) {
			return( glob.Dats.ApCanon[ me.Kparentp ].GetVar(glob, va[1:], lno) );
		}
	}
	if va[0] == "previous" { // one.unit:161, go-struct-rio.act:178
		if (me.Me > 0 && len(va) > 1) {
			return( glob.Dats.ApSection[ me.Me - 1 ].GetVar(glob, va[1:], lno) )
		}
	}
	if len(va) > 1 {
		msg := fmt.Sprintf("?%s.?:%s,%s,Section > one.unit:161, go-struct-rio.act:185?", va[0], lno, me.LineNo)
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
		rr := fmt.Sprintf("?%s?:%s,%s,Section > one.unit:161, go-struct-rio.act:197?", va[0], lno, me.LineNo) 
		return false,rr
	}
	rr := me.Names[va[0]].(string)
	return true,rr
}

func (me KpSection) DoIts(glob *GlobT, va []string, lno string) int {
	if va[0] == "parent" { // one.unit:141, go-struct-rio.act:570
		if me.Kparentp >= 0 {
			st := glob.Dats.ApCanon[ me.Kparentp ]
			if len(va) > 1 {
				return( st.DoIts(glob, va[1:], lno) )
			}
			return( GoAct(glob, st) )
		}
		return(0)
	}
		if va[0] == "level" && len(va) > 1 && me.Kparentp >= 0 { // one.unit:166, go-struct-rio.act:233
			pos := 0
			v, ok := me.Names["level"].(string)
			if ok {
				pos, _ = strconv.Atoi(v)
			}
			if va[1] == "down" && pos > 0 {
				pst := glob.Dats.ApCanon[me.Kparentp]
				isin := false
				for _, st := range pst.ItsSection {
					if st.Me == me.Me {
						isin = true
						continue
					}
					if !isin {
						continue
					}
					pos2 := 0
					v2, ok2 := st.Names["level"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == (pos - 1) {
						break
					}
					if pos2 == pos {
						if len(va) > 2 {
							return st.DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, st)
					}
				}
				return 0
			}
			if va[1] == "up" && pos > 0 {
				pst := glob.Dats.ApCanon[me.Kparentp]
				isin := false
				prev := 0
				for _, st := range pst.ItsSection {
					pos2 := 0
					v2, ok2 := st.Names["level"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == pos && st.Me != me.Me {
						prev = st.Me
						isin = true
						continue
					}
					if pos2 == (pos - 1) {
						isin = false
					}
					if st.Me == me.Me && isin {
						if len(va) > 2 {
							return glob.Dats.ApSection[prev].DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, glob.Dats.ApSection[prev])
					}
				}
				return 0
			}
			if va[1] == "left" && pos > 0 {
				pst := glob.Dats.ApCanon[me.Kparentp]
				isin := false
				prev := 0
				for _, st := range pst.ItsSection {
					pos2 := 0
					v2, ok2 := st.Names["level"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 == (pos - 1) {
						prev = st.Me
						isin = true
						continue
					}
					if st.Me == me.Me && isin {
						if len(va) > 2 {
							return glob.Dats.ApSection[prev].DoIts(glob, va[2:], lno)
						}
						return GoAct(glob, glob.Dats.ApSection[prev])
					}
				}
				return 0
			}
			if va[1] == "right" && pos > 0 {
				pst := glob.Dats.ApCanon[me.Kparentp]
				isin := false
				for _, st := range pst.ItsSection {
					if st.Me == me.Me {
						isin = true
						continue
					}
					if !isin {
						continue
					}
					pos2 := 0
					v2, ok2 := st.Names["level"].(string)
					if ok2 {
						pos2, _ = strconv.Atoi(v2)
					}
					if pos2 == 0 {
						continue
					}
					if pos2 <= pos {
						break
					}
					if pos2 == (pos + 1) {
						if len(va) > 2 {
							ret := st.DoIts(glob, va[2:], lno)
							if ret != 0 {
								return ret
							}
							continue
						}
						ret := GoAct(glob, st)
						if ret != 0 {
							return ret
						}
					}
				}
				return 0
			}
		}
	        fmt.Printf("?No its %s for Section %s,%s > one.unit:161, go-struct-rio.act:222?", va[0], lno, me.LineNo)
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

