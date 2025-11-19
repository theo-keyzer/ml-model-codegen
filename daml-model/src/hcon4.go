package main

import (
	"bufio"
	"os"
	"regexp"
	"strings"
)

// Component represents a parsed block from the ML file
type Component struct {
	Type       string
	Name       string
	Fields     map[string]string
	LineNumber int // Line where this component starts
}

// stripComments removes inline comments (//) from a string, 
// assuming it's not inside a multi-line string.
// This is a simplified function and assumes // is not part of a quoted field value.
func stripComments(line string) string {
	if idx := strings.Index(line, "//"); idx != -1 {
		return strings.TrimSpace(line[:idx])
	}
	return strings.TrimSpace(line)
}

// ParseMLFile reads the ML format and returns structured components
func ParseMLFile(filename string) ([]Component, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var components []Component
	scanner := bufio.NewScanner(file)
	
	var currentComp *Component
	var inBlock bool
	var inMultiLine bool
	var multiLineKey string
	var multiLineDelimiter string // Track which delimiter (""" or ~~~) we'sre using
	var multiLineContent strings.Builder
	var blockType, blockName string
	lineNumber := 0

	for scanner.Scan() {
		lineNumber++
		line := strings.TrimRight(scanner.Text(), " \t") // Trim trailing whitespace

		// Handle multi-line string content
		if inMultiLine {
			trimmed := strings.TrimSpace(line)
			// Check if this line ends the multi-line block
			if trimmed == multiLineDelimiter {
				// End of multi-line string
				inMultiLine = false
				if currentComp != nil {
					// Trim trailing newline added by the builder
					currentComp.Fields[multiLineKey] = strings.TrimRight(multiLineContent.String(), "\n")
				}
				multiLineContent.Reset()
				multiLineDelimiter = ""
				continue
			} else {
				// Add line to multi-line content
				multiLineContent.WriteString(line)
				multiLineContent.WriteString("\n")
				continue
			}
		}
		
		// --- REVISED LOGIC FOR SKIPPING LINES ---
		// Skip fully empty lines
		if line == "" {
			continue
		}
		// Skip full-line comments outside of a block
		if !inBlock && strings.HasPrefix(strings.TrimSpace(line), "//") {
			continue
		}
		// ----------------------------------------

		// Check for start of multi-line string (""" or ~~~)
		if !inMultiLine {
			// First, process the line to remove any trailing comment *if* it starts a multi-line block.
			processedLine := line
			if strings.Contains(line, "//") {
				processedLine = stripComments(line)
			}

			for _, delimiter := range []string{`"""`, `~~~`} {
				if strings.Contains(processedLine, delimiter) {
					parts := strings.SplitN(processedLine, delimiter, 2)
					if len(parts) >= 1 {
						// Find the key: look for the part before the delimiter
						beforeDelimiter := strings.TrimSpace(parts[0])
						
						// Use a simple split by '=' to isolate the key name
						if lastEqual := strings.LastIndex(beforeDelimiter, "="); lastEqual != -1 {
							// The key is the content before the last '='
							keyCandidate := strings.TrimSpace(beforeDelimiter[:lastEqual])
							
							// The key name should be the last word in keyCandidate
							keyParts := strings.Fields(keyCandidate)
							if len(keyParts) > 0 {
								multiLineKey = keyParts[len(keyParts)-1]
								inMultiLine = true
								multiLineDelimiter = delimiter
								
								// Check if there's content on the same line after delimiter
								if len(parts) > 1 {
									afterDelimiter := parts[1]
									// Check if it also ends on the same line
									if strings.Contains(afterDelimiter, delimiter) {
										// Single-line multi-line block (e.g., key = """value""")
										endParts := strings.SplitN(afterDelimiter, delimiter, 2)
										if currentComp != nil {
											currentComp.Fields[multiLineKey] = endParts[0]
										}
										inMultiLine = false
										multiLineDelimiter = ""
									} else if afterDelimiter != "" {
										multiLineContent.WriteString(afterDelimiter)
										multiLineContent.WriteString("\n")
									}
								}
								break
							}
						}
					}
				}
			}
			if inMultiLine {
				continue
			}
		}

		// Start of a new component block
		if !inMultiLine && strings.HasSuffix(strings.TrimSpace(line), "{") {
			parts := strings.Fields(strings.TrimSpace(line))
			if len(parts) >= 2 {
				blockType = parts[0]
				blockName = strings.TrimSuffix(parts[1], "{")
				// Remove quotes from name if present
				blockName = strings.Trim(blockName, `"`)
				currentComp = &Component{
					Type:       blockType,
					Name:       blockName,
					Fields:     make(map[string]string),
					LineNumber: lineNumber,
				}
				inBlock = true
			}
			continue
		}

		// End of component block
		if strings.TrimSpace(line) == "}" && inBlock {
			if currentComp != nil {
				components = append(components, *currentComp)
			}
			inBlock = false
			currentComp = nil
			continue
		}

		// Inside component block - parse fields
		if inBlock && currentComp != nil {
			parseField(line, currentComp)
		}
	}

	return components, scanner.Err()
}

// parseField extracts key/value pairs from a line.
// This function is modified to:
// 1. Trim whitespace and comments for simple key=value fields.
// 2. Only trim whitespace for list fields (key=[v1, v2]).
func parseField(line string, comp *Component) {
	// Trim leading/trailing whitespace and comments from the line for consistent regex matching
	processedLine := stripComments(line)

	// Skip empty lines after processing
	if processedLine == "" {
		return
	}

	// Handle different field patterns
	patterns := []struct {
		regex  *regexp.Regexp
		isList bool
	}{
		// Array field: inputs = [value1, value2]
		// Captures the key and the content inside the brackets.
		{regexp.MustCompile(`^(\w+)\s*=\s*\[(.+)\]$`), true},
		// Simple key-value: name = value
		// Captures the key and the raw value content.
		{regexp.MustCompile(`^(\w+)\s*=\s*(.+)$`), false},
	}

	for _, pattern := range patterns {
		if pattern.regex.MatchString(processedLine) {
			matches := pattern.regex.FindStringSubmatch(processedLine)
			
			key := matches[1]
			rawValue := matches[2] // Content after '=' (or inside brackets for a list)

			if pattern.isList {
				// Array field: Only trim leading/trailing whitespace from the content 
				// captured inside the brackets. This fixes the issue.
				comp.Fields[key] = strings.TrimSpace(rawValue)
			} else {
				// Simple key-value: Trim whitespace and remove quotes.
				value := strings.TrimSpace(rawValue)
				
				// Remove quotes if present
				value = strings.Trim(value, `"`)
				
				// Don't process if it's a multi-line string starter (already handled in ParseMLFile)
				if !strings.Contains(value, `"""`) && !strings.Contains(value, `~~~`) {
					comp.Fields[key] = value
				}
			}
			return
		}
	}
}

