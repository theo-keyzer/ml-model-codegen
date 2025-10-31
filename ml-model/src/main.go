package main

import (
	"fmt"
	"os"
	"strings"
)

// GlobT represents global state
type GlobT struct {
	LoadErrs  int
	RunErrs   int
	Acts      ActT
	Dats      ActT
	Winp      int
	Wins      []WinT
	Collect   map[string]interface{}
	OutOn     bool
	InOn      bool
	Ins       strings.Builder
}


// KpExtra represents extra key-value pairs
type KpExtra struct {
	Kp
	Names map[string]string
}
func (me KpExtra) GetVar(glob *GlobT, s []string, ln string) (bool, string) {
	r,ok := me.Names[s[0]]
//	if !ok { r = fmt.Sprintf("?%s?:%s, Command line arguments", s[0], ln) }
	if !ok { r = fmt.Sprintf("xxx") }
	return ok,r
}

func (me KpExtra) GetLineNo(glob *GlobT, s []string, ln string) (string) {
	return "Command line arguments"
}

func main() {
	args := os.Args[1:] // Skip the program name
	if len(args) < 2 {
		fmt.Println(args)
		return
	}

	glob := new(GlobT)
	glob.Winp = -1
	glob.OutOn = true
	glob.Collect = make(map[string]interface{})
	// Load files and check for errors
	glob.LoadErrs += loadFiles(args[0], &glob.Acts)
	glob.LoadErrs += loadFilesh(args[1], &glob.Dats)

	if len(glob.Acts.ApActor) > 0 {
		kp := &KpExtra{
			Names: make(map[string]string),
		}
		
		// Store args in kp.Names
		for i, arg := range args {
			kp.Names[fmt.Sprint(i)] = arg
		}
		NewAct(glob, glob.Acts.ApActor[0].Kname, "", "run:1")
		GoAct(glob, kp)
	}

	if glob.LoadErrs > 0 || glob.RunErrs > 0 {
//		fmt.Println("Errors", glob.LoadErrs, glob.RunErrs)
		println("Errors", glob.LoadErrs, glob.RunErrs)
		os.Exit(1)
	}
}

func loadFilesh(files string, act *ActT) int {
	act.index = make(map[string]int) 
	errs := 0
	fileList := strings.Split(files, ",")
	
	for _, file := range fileList {
		components, err := ParseMLFile(file)
		if err != nil {
			fmt.Printf("Error reading h file %s: %v\n", file, err)
			errs = errs + 1
			continue
		}
		for _, comp := range components {
			lno := fmt.Sprintf("%s:%d", file, comp.LineNumber)
			Loadh(act, comp.Type, "", 7, lno,comp.Fields)
		}
		// components, err := ParseMLFile("ml_ml.txt")
		/*
		content, err := os.ReadFile(file)
		if err != nil {
			fmt.Printf("Error reading file %s: %v\n", file, err)
			errs = errs + 1
			continue
		}
		*/
//		Loadh(act, "", "", 7, "v:8",components)
//		lines := strings.Split(string(content), "\n")
//		errs += LoadDatah(lines, act, file,components)
	}
	
	errs += refs(act)
	return errs
}
func loadFiles(files string, act *ActT) int {
	act.index = make(map[string]int) 
	errs := 0
	fileList := strings.Split(files, ",")
	
	for _, file := range fileList {
		content, err := os.ReadFile(file)
		if err != nil {
			fmt.Printf("Error reading file %s: %v\n", file, err)
			errs = errs + 1
			continue
		}

		lines := strings.Split(string(content), "\n")
		errs += LoadData(lines, act, file)
	}
	
	errs += refs(act)
	return errs
}

// LoadData loads data from lines into an actor
func LoadData(lns []string, act *ActT, file string) int {
	errs := 0
	for i := 0; i < len(lns); i++ {
		lno := fmt.Sprintf("%s:%d", file, i+1)
		line := strings.ReplaceAll(lns[i], "\r", "")
		pos,tok := getw( line, 0)
		if tok == "E_O_F" {
			break
		}
		errs += Load(act, tok, line, pos, lno)
	}
	return errs
}

// fnd3 finds a value in the index
func fnd3(act *ActT, s string, f string, msg string, chk string, lno string, olno string) (bool, int) {
	if v, exists := act.index[s]; exists {
		return true, v
	}
	
	if chk == "*" || chk == "?" {
		return true, -1
	}
	if f == chk {
		return true, -1
	}
	if f == "" && chk == "." {
		return true, -1
	}
	
	fmt.Printf("%s %s (%s) not found %s (%s) > %s\n", msg,f, s, lno,chk,olno)
	return false, -1
}
// fnd2 finds a value in the index
func fnd2(act *ActT, s string, f string, chk string, lno string, olno string) (bool, int) {
	if v, exists := act.index[s]; exists {
		return true, v
	}
	
	if chk == "*" || chk == "?" {
		return true, -1
	}
	if f == chk {
		return true, -1
	}
	
	fmt.Printf("%s (%s) not found %s (%s) > %s\n", f, s, lno,chk,olno)
	return false, -1
}
func fnd(act *ActT, s string, f string, chk string, lno string) (bool, int) {
	if v, exists := act.index[s]; exists {
		return true, v
	}
	
	if chk == "?" {
		return true, -1
	}
	if f == chk {
		return true, -1
	}
	
	fmt.Printf("%s (%s) not found %s (%s)\n", f, s, lno,chk)
	return false, -1
}

// getName gets a value from a map by name
func getName(m map[string]string, n string) string {
	if v, ok := m[n]; ok {
		return v
	}
	return ""
}

// getws gets the rest of the line after position
func getws(line string, pos int) (int, string) {
	if pos+1 > len(line) {
		return pos, ""
	}
	return len(line), line[pos+1:]
}

// getw gets the next word
func getw(line string, pos int) (int, string) {
	if pos+1 > len(line) {
		return pos, "E_O_L"
	}

	// Skip whitespace
	from := pos
	for i := pos; i < len(line); i++ {
		if line[i] != ' ' && line[i] != '\t' {
			from = i
			break
		}
	}

	if from+1 > len(line) {
		return pos, "E_O_L"
	}

	// Find word end
	to := from
	for i := from; i < len(line); i++ {
		if line[i] == ' ' || line[i] == '\t' {
			break
		}
		to = i
	}

	return to + 1, line[from : to+1]
}

// getsw gets the next word that might be followed by a colon
func getsw(line string, pos int) (int, string) {
	if pos+1 > len(line) {
		return pos, "E_O_L"
	}

	// Skip whitespace
	from := pos
	for i := pos; i < len(line); i++ {
		if line[i] != ' ' && line[i] != '\t' {
			from = i
			break
		}
	}

	if from+1 > len(line) {
		return pos, "E_O_L"
	}

	// Find word end
	to := from
	isS := false
	for i := from; i < len(line); i++ {
		if line[i] == ' ' || line[i] == '\t' {
			break
		}
		if line[i] == ':' {
			isS = true
			break
		}
		to = i
	}

	if isS {
		return to + 2, line[from : to+1]
	}
	return to + 1, line[from : to+1]
}

