//! Security testing and vulnerability scanning for photonic simulation

use super::*;
use crate::core::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use regex::Regex;

/// Security scanner for detecting common vulnerabilities
pub struct SecurityScanner {
    rules: Vec<SecurityRule>,
    scan_config: ScanConfig,
}

impl SecurityScanner {
    pub fn new() -> Self {
        let mut scanner = Self {
            rules: Vec::new(),
            scan_config: ScanConfig::default(),
        };
        
        scanner.load_default_rules();
        scanner
    }
    
    pub fn with_config(config: ScanConfig) -> Self {
        let mut scanner = Self::new();
        scanner.scan_config = config;
        scanner
    }
    
    pub fn scan_directory<P: AsRef<Path>>(&self, path: P) -> Result<SecurityReport> {
        let start_time = std::time::Instant::now();
        let mut report = SecurityReport::new();
        
        let files = self.collect_files(path.as_ref())?;
        report.files_scanned = files.len();
        
        for file_path in files {
            if let Ok(content) = fs::read_to_string(&file_path) {
                report.lines_scanned += content.lines().count();
                
                for rule in &self.rules {
                    let vulnerabilities = rule.check(&content, &file_path)?;
                    report.vulnerabilities.extend(vulnerabilities);
                }
            }
        }
        
        report.scan_time = start_time.elapsed();
        Ok(report)
    }
    
    fn load_default_rules(&mut self) {
        // Hardcoded credentials detection
        self.rules.push(SecurityRule {
            id: "SEC001".to_string(),
            name: "Hardcoded Credentials".to_string(),
            description: "Detects hardcoded passwords, API keys, and secrets".to_string(),
            pattern: Regex::new(r#"(?i)(password|pwd|secret|key|token|auth)\s*[:=]\s*["'][^"']{8,}["']"#).unwrap(),
            severity: Severity::High,
            category: "Credential Management".to_string(),
        });
        
        // SQL Injection patterns
        self.rules.push(SecurityRule {
            id: "SEC002".to_string(),
            name: "Potential SQL Injection".to_string(),
            description: "Detects unsafe SQL query construction".to_string(),
            pattern: Regex::new(r#"(?i)(execute|query|select|insert|update|delete)\s*\([^)]*\+[^)]*\)"#).unwrap(),
            severity: Severity::Medium,
            category: "Injection".to_string(),
        });
        
        // Unsafe memory operations
        self.rules.push(SecurityRule {
            id: "SEC003".to_string(),
            name: "Unsafe Memory Operations".to_string(),
            description: "Detects potentially unsafe memory operations".to_string(),
            pattern: Regex::new(r#"\bunsafe\s*\{[^}]*\breturn\b[^}]*\}"#).unwrap(),
            severity: Severity::Medium,
            category: "Memory Safety".to_string(),
        });
        
        // Debug information leakage
        self.rules.push(SecurityRule {
            id: "SEC004".to_string(),
            name: "Debug Information Exposure".to_string(),
            description: "Detects debug information that could leak sensitive data".to_string(),
            pattern: Regex::new(r#"(println!|eprintln!|dbg!)\s*\([^)]*(?:password|secret|key|token)[^)]*\)"#).unwrap(),
            severity: Severity::Low,
            category: "Information Disclosure".to_string(),
        });
        
        // Insecure random number generation
        self.rules.push(SecurityRule {
            id: "SEC005".to_string(),
            name: "Weak Random Number Generation".to_string(),
            description: "Detects use of predictable random number generators".to_string(),
            pattern: Regex::new(r#"rand::random\(\)|thread_rng\(\)"#).unwrap(),
            severity: Severity::Low,
            category: "Cryptography".to_string(),
        });
        
        // Unvalidated input
        self.rules.push(SecurityRule {
            id: "SEC006".to_string(),
            name: "Unvalidated External Input".to_string(),
            description: "Detects direct use of external input without validation".to_string(),
            pattern: Regex::new(r#"env::args\(\)\..*\.unwrap\(\)"#).unwrap(),
            severity: Severity::Medium,
            category: "Input Validation".to_string(),
        });
        
        // Buffer overflow risks
        self.rules.push(SecurityRule {
            id: "SEC007".to_string(),
            name: "Buffer Overflow Risk".to_string(),
            description: "Detects operations that could lead to buffer overflows".to_string(),
            pattern: Regex::new(r#"get_unchecked\(|slice_unchecked\("#).unwrap(),
            severity: Severity::High,
            category: "Memory Safety".to_string(),
        });
        
        // Cryptographic weaknesses
        self.rules.push(SecurityRule {
            id: "SEC008".to_string(),
            name: "Weak Cryptographic Algorithm".to_string(),
            description: "Detects use of weak or deprecated cryptographic algorithms".to_string(),
            pattern: Regex::new(r#"(?i)(md5|sha1|des|rc4)\b"#).unwrap(),
            severity: Severity::Medium,
            category: "Cryptography".to_string(),
        });
    }
    
    fn collect_files(&self, path: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        
        if path.is_file() {
            if self.should_scan_file(path) {
                files.push(path.to_path_buf());
            }
        } else if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() && !self.should_skip_directory(&path) {
                    files.extend(self.collect_files(&path)?);
                } else if self.should_scan_file(&path) {
                    files.push(path);
                }
            }
        }
        
        Ok(files)
    }
    
    fn should_scan_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
            self.scan_config.file_extensions.contains(&extension.to_string())
        } else {
            false
        }
    }
    
    fn should_skip_directory(&self, path: &Path) -> bool {
        if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
            self.scan_config.excluded_directories.contains(&dir_name.to_string())
        } else {
            false
        }
    }
}

/// Security scanning rule
#[derive(Debug, Clone)]
pub struct SecurityRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub pattern: Regex,
    pub severity: Severity,
    pub category: String,
}

impl SecurityRule {
    pub fn check(&self, content: &str, file_path: &Path) -> Result<Vec<Vulnerability>> {
        let mut vulnerabilities = Vec::new();
        
        for (line_no, line) in content.lines().enumerate() {
            if self.pattern.is_match(line) {
                vulnerabilities.push(Vulnerability {
                    id: self.id.clone(),
                    description: format!("{}: {}", self.name, self.description),
                    severity: self.severity.clone(),
                    file_path: Some(file_path.to_string_lossy().to_string()),
                    line_number: Some(line_no + 1),
                    recommendation: self.get_recommendation(),
                });
            }
        }
        
        Ok(vulnerabilities)
    }
    
    fn get_recommendation(&self) -> String {
        match self.id.as_str() {
            "SEC001" => "Use environment variables or secure configuration management for credentials".to_string(),
            "SEC002" => "Use parameterized queries or prepared statements to prevent SQL injection".to_string(),
            "SEC003" => "Review unsafe blocks carefully and minimize their use".to_string(),
            "SEC004" => "Avoid logging sensitive information in debug statements".to_string(),
            "SEC005" => "Use cryptographically secure random number generators for security-critical operations".to_string(),
            "SEC006" => "Validate and sanitize all external input before processing".to_string(),
            "SEC007" => "Use safe indexing methods and bounds checking".to_string(),
            "SEC008" => "Replace deprecated algorithms with modern, secure alternatives".to_string(),
            _ => "Review the flagged code for potential security issues".to_string(),
        }
    }
}

/// Security scan configuration
#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub file_extensions: Vec<String>,
    pub excluded_directories: Vec<String>,
    pub max_file_size: usize,
    pub enable_deep_scan: bool,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            file_extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "toml".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
                "json".to_string(),
            ],
            excluded_directories: vec![
                "target".to_string(),
                ".git".to_string(),
                "node_modules".to_string(),
                "__pycache__".to_string(),
                ".pytest_cache".to_string(),
            ],
            max_file_size: 1024 * 1024, // 1MB
            enable_deep_scan: true,
        }
    }
}

/// Dependency vulnerability scanner
pub struct DependencyScanner {
    known_vulnerabilities: HashMap<String, Vec<KnownVulnerability>>,
}

impl DependencyScanner {
    pub fn new() -> Self {
        let mut scanner = Self {
            known_vulnerabilities: HashMap::new(),
        };
        
        scanner.load_vulnerability_database();
        scanner
    }
    
    pub fn scan_dependencies(&self, manifest_path: &Path) -> Result<Vec<Vulnerability>> {
        let mut vulnerabilities = Vec::new();
        
        if manifest_path.file_name().and_then(|s| s.to_str()) == Some("Cargo.toml") {
            vulnerabilities.extend(self.scan_cargo_dependencies(manifest_path)?);
        } else if manifest_path.file_name().and_then(|s| s.to_str()) == Some("requirements.txt") {
            vulnerabilities.extend(self.scan_python_dependencies(manifest_path)?);
        }
        
        Ok(vulnerabilities)
    }
    
    fn scan_cargo_dependencies(&self, cargo_toml: &Path) -> Result<Vec<Vulnerability>> {
        let mut vulnerabilities = Vec::new();
        
        if let Ok(content) = fs::read_to_string(cargo_toml) {
            // Simple regex-based parsing (in production, use proper TOML parser)
            let dep_regex = Regex::new(r#"^([a-zA-Z0-9_-]+)\s*=\s*["']([^"']+)["']"#)?;
            
            for line in content.lines() {
                if let Some(captures) = dep_regex.captures(line) {
                    let package_name = captures.get(1).unwrap().as_str();
                    let version = captures.get(2).unwrap().as_str();
                    
                    if let Some(known_vulns) = self.known_vulnerabilities.get(package_name) {
                        for vuln in known_vulns {
                            if vuln.affects_version(version) {
                                vulnerabilities.push(Vulnerability {
                                    id: vuln.id.clone(),
                                    description: format!("Vulnerable dependency: {} {} - {}", 
                                                       package_name, version, vuln.description),
                                    severity: vuln.severity.clone(),
                                    file_path: Some(cargo_toml.to_string_lossy().to_string()),
                                    line_number: None,
                                    recommendation: format!("Update {} to version {} or later", 
                                                          package_name, vuln.fixed_version),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(vulnerabilities)
    }
    
    fn scan_python_dependencies(&self, requirements_txt: &Path) -> Result<Vec<Vulnerability>> {
        let mut vulnerabilities = Vec::new();
        
        if let Ok(content) = fs::read_to_string(requirements_txt) {
            let dep_regex = Regex::new(r#"^([a-zA-Z0-9_-]+)==([0-9.]+)"#)?;
            
            for line in content.lines() {
                if let Some(captures) = dep_regex.captures(line) {
                    let package_name = captures.get(1).unwrap().as_str();
                    let version = captures.get(2).unwrap().as_str();
                    
                    if let Some(known_vulns) = self.known_vulnerabilities.get(package_name) {
                        for vuln in known_vulns {
                            if vuln.affects_version(version) {
                                vulnerabilities.push(Vulnerability {
                                    id: vuln.id.clone(),
                                    description: format!("Vulnerable dependency: {} {} - {}", 
                                                       package_name, version, vuln.description),
                                    severity: vuln.severity.clone(),
                                    file_path: Some(requirements_txt.to_string_lossy().to_string()),
                                    line_number: None,
                                    recommendation: format!("Update {} to version {} or later", 
                                                          package_name, vuln.fixed_version),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(vulnerabilities)
    }
    
    fn load_vulnerability_database(&mut self) {
        // In production, this would load from a real vulnerability database
        // For demo purposes, we'll add some simulated vulnerabilities
        
        self.known_vulnerabilities.insert("serde".to_string(), vec![
            KnownVulnerability {
                id: "CVE-2021-XXXX".to_string(),
                description: "Hypothetical deserialization vulnerability".to_string(),
                severity: Severity::High,
                affected_versions: vec!["<1.0.130".to_string()],
                fixed_version: "1.0.130".to_string(),
            }
        ]);
        
        self.known_vulnerabilities.insert("numpy".to_string(), vec![
            KnownVulnerability {
                id: "CVE-2020-YYYY".to_string(),
                description: "Buffer overflow in array handling".to_string(),
                severity: Severity::Medium,
                affected_versions: vec!["<1.19.0".to_string()],
                fixed_version: "1.19.0".to_string(),
            }
        ]);
    }
}

/// Known vulnerability information
#[derive(Debug, Clone)]
pub struct KnownVulnerability {
    pub id: String,
    pub description: String,
    pub severity: Severity,
    pub affected_versions: Vec<String>,
    pub fixed_version: String,
}

impl KnownVulnerability {
    pub fn affects_version(&self, version: &str) -> bool {
        // Simplified version comparison (in production, use proper version parsing)
        for affected_pattern in &self.affected_versions {
            if affected_pattern.starts_with('<') {
                let threshold = &affected_pattern[1..];
                if version < threshold {
                    return true;
                }
            } else if affected_pattern == version {
                return true;
            }
        }
        false
    }
}

/// Comprehensive security test runner
pub async fn run_security_tests(project_root: &Path) -> Result<SecurityReport> {
    println!("🔒 Running comprehensive security tests...");
    
    let mut combined_report = SecurityReport::new();
    let start_time = std::time::Instant::now();
    
    // 1. Static code analysis
    println!("📋 Performing static code analysis...");
    let scanner = SecurityScanner::new();
    let static_report = scanner.scan_directory(project_root.join("src"))?;
    
    combined_report.vulnerabilities.extend(static_report.vulnerabilities);
    combined_report.files_scanned += static_report.files_scanned;
    combined_report.lines_scanned += static_report.lines_scanned;
    
    // 2. Dependency vulnerability scanning
    println!("📦 Scanning dependencies for known vulnerabilities...");
    let dep_scanner = DependencyScanner::new();
    
    let cargo_toml = project_root.join("Cargo.toml");
    if cargo_toml.exists() {
        let dep_vulnerabilities = dep_scanner.scan_dependencies(&cargo_toml)?;
        combined_report.vulnerabilities.extend(dep_vulnerabilities);
    }
    
    let requirements_txt = project_root.join("requirements.txt");
    if requirements_txt.exists() {
        let dep_vulnerabilities = dep_scanner.scan_dependencies(&requirements_txt)?;
        combined_report.vulnerabilities.extend(dep_vulnerabilities);
    }
    
    // 3. Configuration security check
    println!("⚙️ Checking configuration security...");
    let config_vulnerabilities = check_configuration_security(project_root)?;
    combined_report.vulnerabilities.extend(config_vulnerabilities);
    
    combined_report.scan_time = start_time.elapsed();
    
    println!("🔒 Security scan completed: {}", combined_report.summary());
    
    Ok(combined_report)
}

/// Check configuration files for security issues
fn check_configuration_security(project_root: &Path) -> Result<Vec<Vulnerability>> {
    let mut vulnerabilities = Vec::new();
    
    // Check for exposed debug/development configurations
    let config_patterns = vec![
        (project_root.join("Cargo.toml"), r#"debug\s*=\s*true"#),
        (project_root.join(".env"), r#"DEBUG\s*=\s*(true|1)"#),
    ];
    
    for (config_file, pattern) in config_patterns {
        if config_file.exists() {
            if let Ok(content) = fs::read_to_string(&config_file) {
                let regex = Regex::new(pattern)?;
                if regex.is_match(&content) {
                    vulnerabilities.push(Vulnerability {
                        id: "SEC-CONFIG-001".to_string(),
                        description: "Debug mode enabled in configuration".to_string(),
                        severity: Severity::Medium,
                        file_path: Some(config_file.to_string_lossy().to_string()),
                        line_number: None,
                        recommendation: "Disable debug mode in production builds".to_string(),
                    });
                }
            }
        }
    }
    
    Ok(vulnerabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_security_rule_detection() -> Result<()> {
        let rule = SecurityRule {
            id: "TEST001".to_string(),
            name: "Test Rule".to_string(),
            description: "Test hardcoded password detection".to_string(),
            pattern: Regex::new(r#"password\s*[:=]\s*["'][^"']+["']"#)?,
            severity: Severity::High,
            category: "Test".to_string(),
        };
        
        let test_content = r#"
            let username = "admin";
            let password = "secret123"; // This should be detected
            let config_file = "/etc/config";
        "#;
        
        let temp_file = std::path::Path::new("test.rs");
        let vulnerabilities = rule.check(test_content, temp_file)?;
        
        assert_eq!(vulnerabilities.len(), 1);
        assert_eq!(vulnerabilities[0].severity, Severity::High);
        
        Ok(())
    }
    
    #[test]
    fn test_dependency_vulnerability_detection() {
        let mut scanner = DependencyScanner::new();
        scanner.known_vulnerabilities.insert("test-package".to_string(), vec![
            KnownVulnerability {
                id: "TEST-CVE-001".to_string(),
                description: "Test vulnerability".to_string(),
                severity: Severity::High,
                affected_versions: vec!["<2.0.0".to_string()],
                fixed_version: "2.0.0".to_string(),
            }
        ]);
        
        // Test version comparison
        let vuln = &scanner.known_vulnerabilities["test-package"][0];
        assert!(vuln.affects_version("1.5.0"));
        assert!(!vuln.affects_version("2.0.0"));
        assert!(!vuln.affects_version("2.1.0"));
    }
    
    #[tokio::test]
    async fn test_security_scan_integration() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let temp_path = temp_dir.path();
        
        // Create a test source file with a vulnerability
        let src_dir = temp_path.join("src");
        fs::create_dir_all(&src_dir)?;
        
        let test_file = src_dir.join("test.rs");
        fs::write(&test_file, r#"
            fn main() {
                let password = "hardcoded_secret_123";
                println!("Password: {}", password);
            }
        "#)?;
        
        // Run security scan
        let report = run_security_tests(temp_path).await?;
        
        assert!(report.vulnerabilities.len() > 0, "Should detect vulnerabilities");
        assert!(report.files_scanned > 0, "Should scan files");
        
        println!("Security test report: {}", report.summary());
        
        Ok(())
    }
}