#!/usr/bin/env python3
"""
Sequential HR Questions Test Script for FlowHCM RAG Chatbot
Tests specific HR-related questions one by one and documents retrieval details.
"""

import asyncio
import aiohttp
import json
import time
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
import statistics
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TimingMetrics:
    """Detailed timing metrics for each operation"""
    health_check_time: float = 0.0
    db_auth_time: float = 0.0
    request_preparation_time: float = 0.0
    network_request_time: float = 0.0
    response_processing_time: float = 0.0
    log_parsing_time: float = 0.0
    total_operation_time: float = 0.0
    
    # RAG-specific timings (from logs)
    rag_retrieval_time: float = 0.0
    rag_generation_time: float = 0.0
    rag_total_processing_time: float = 0.0
    
    # Database operation timings (estimated from logs)
    db_session_lookup_time: float = 0.0
    db_session_save_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

# Test Configuration
API_BASE_URL = "http://localhost:8000"
TEST_RESULTS_FILE = "hr_questions_test_results.json"

# Test credentials
TEST_COMPANY_PIN = "11032"
TEST_API_KEY = "test123456789012345678901234567890"
TEST_USER_NAME = "HR_Tester"
TEST_USER_ID = "hr_test_001"

# HR Questions to test - Mixed comprehensive list
HR_QUESTIONS = [
    # Employee Exit Process
    "What are the steps to apply for an employee exit, and how can clearance and exit interviews be enabled during the exit request?",
    "Write a job description (JD) for a Software Engineer.",
    "How can bulk exit requests be submitted using Excel, and what common issues could prevent the upload?",
    "Create Key Performance Indicators (KPIs) for a Product Owner.",
    "What is the process for running a final settlement, and how can the FnF details be previewed and disbursed?",
    "Generate a leave policy for annual, sick, and casual leaves.",
    "How does the Employee Clearance workflow function, and what could cause a clearance request not to appear in the grid?",
    "Draft an employee onboarding checklist for new hires.",
    "What steps are involved in conducting an exit interview, and how are templates and interviewer assignments managed?",
    "Prepare a performance appraisal form for managers.",
    
    # Payroll Process Mixed
    "What are the key differences between Payroll Setup and Salary Setup in FlowHCM, and how do they interact?",
    "Design a salary structure for entry-level marketing executives.",
    "How can recurring allowances and deductions be applied to employees, and what options are available for bulk uploads?",
    "Generate a training plan for onboarding remote employees.",
    "What steps are required to process payroll, and how can payroll be previewed before disbursement?",
    "Create an exit interview questionnaire for resigning employees.",
    "Which types of employee requests require approval in the Payroll module, and where can these approvals be managed?",
    "Draft a company-wide attendance policy for hybrid work.",
    "How is overtime configured and requested, and what conditions could prevent it from appearing in payroll?",
    "Suggest a bonus policy for top-performing sales staff.",
    
    # Employee Management Mixed
    "What is the difference between adding an employee through Employee List and submitting an Employee Profile Request?",
    "Prepare a workflow for advance salary requests.",
    "Which sections must be completed when adding a new employee directly, and which section is excluded in approval-based onboarding?",
    "Create a reimbursement policy for travel and meals.",
    "How can a previously resigned employee be rehired, and what could prevent the rehire from succeeding?",
    "Generate a provident fund (PF) contribution schedule for employees.",
    "What is the difference between a Transfer and a Transition in the Employee Transfer module?",
    "Draft a standard operating procedure (SOP) for employee promotions.",
    "How can roles and permissions be assigned to multiple employees without configuring each one manually?",
    "Prepare a multi-level approval hierarchy for leave and payroll requests.",
    
    # Leave Management Mixed
    "What steps must an employee follow to submit a leave request in the Leave Request module?",
    "How are employee separation approvals handled, and how does the system manage multi-level approval hierarchies?",
    "How does the system handle overlapping leave dates when submitting a new leave request?",
    "What are the key configuration options in Employee Separation Settings, and how do simultaneous versus sequential approvals work?",
    "What information is required when applying for leave via the bulk Excel upload?",
    "How can a Clearance Template be created, and what requirements must be met to avoid errors during saving?",
    "How can an employee request compensation leave for overtime work?",
    "How are Questionnaire Categories used in exit interviews, and what could prevent a category from saving correctly?",
    "What validations are performed before a compensation leave request is submitted?",
    "How are Exit Interview Templates configured, and what steps ensure questions are properly mapped to categories to avoid duplication errors?",
    
    # Attendance Management Mixed
    "Which types of leave requests can be approved through the Leave Approval module?",
    "How are loans and loan adjustments handled, and what are the differences between general, PF, and gratuity loans?",
    "Why might leave requests not appear for approval in the Leave Approval dashboard?",
    "What is the purpose of VPS and PF policies, and how can issues with their calculation or deduction be resolved?",
    "What configurations can be managed in the General Leave Settings section?",
    "How can salary distribution be configured across multiple stations, and what must be ensured for the distribution to save correctly?",
    "How can leave types be customized in the Leave Type Setting sub-section?",
    "What are the steps for applying for advance salary, leave encashment, or allowance requests, and what policies could block these requests?",
    "How are individual employee leave quotas assigned or modified, and what issues can prevent quota assignment?",
    "How does the system handle tax adjustments, withholding taxes, and fiscal year configurations to ensure compliance?",
    
    # More Mixed Questions
    "What types of attendance regularizations can an employee request through the Attendance Request module?",
    "Which employee information fields can be updated through an Employee Info Request, and how are these updates controlled?",
    "What validations must be satisfied before an attendance request with In Time and Out Time can be submitted?",
    "What conditions must be met for an employee resignation request to be successfully submitted?",
    "Why might an exemption request fail to show available flags for a selected employee and date?",
    "Which types of employee requests can be approved through the Employee Approvals dashboard?",
    "What conditions must be met for a Work Sheet project to appear in the project dropdown?",
    "How does a Delegation Request work, and what issues can prevent it from being active?",
    "What information is mandatory when submitting a Remote Work Request for approval?",
    "How does Amend Employee Dept differ from an employee transfer, and does it require approval?",
    
    # Final Mixed Set
    "Why could a desired shift be unavailable when submitting a Shift Request?",
    "What is the difference between adding an employee through the Employee List and creating one via Employee Profile Request?",
    "What is the difference between applying a penalty and reposting attendance in the Amend Attendance module?",
    "Which sections are mandatory when adding a new employee directly, and which section is excluded in approval-based onboarding?",
    "Why might amended or reposted attendance data not immediately reflect on the dashboard?",
    "What steps should be followed if an Excel bulk upload for employees fails?",
    "What are the possible reasons attendance-related requests are not visible in the Attendance Approvals screen?",
    "Why can a previously resigned employee fail to be rehired even when the CNIC is correct?",
    "What prerequisites must be met for a shift swap or transfer to be successfully submitted in a Shift Transfer Request?",
    "What is the difference between an employee Transfer and a Transition, and what changes are allowed in each?",
    "How can roles and permissions be assigned to multiple employees without configuring each one individually?",
    "Which employee information fields can be updated through an Employee Info Request, and how are these updates controlled?",
    "What conditions must be met for an employee resignation request to be successfully submitted?",
    "Why might an approver not see pending employee requests in the Employee Approvals dashboard?",
    "How does Amend Employee Dept differ from Employee Transfer, and does it require approval?"
]


class HRQuestionsTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        self.log_file_path = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        # Find the most recent log file for the test company
        self.log_file_path = self.find_latest_log_file()
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def find_latest_log_file(self) -> str:
        """Find the latest log file for the test company"""
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs', TEST_COMPANY_PIN)
        if not os.path.exists(logs_dir):
            logger.warning(f"Log directory not found: {logs_dir}")
            return None
        
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(logs_dir, f"{today}.log")
        
        if os.path.exists(log_file):
            logger.info(f"Found log file: {log_file}")
            return log_file
        else:
            logger.warning(f"Log file not found: {log_file}")
            return None
    
    def parse_rag_logs_for_query(self, query_text: str, start_time: datetime) -> Dict[str, Any]:
        """Parse RAG logs to extract detailed information for a specific query"""
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return {"error": "Log file not available"}
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Find the query ID for this specific query
            query_pattern = rf"RAG_QUERY_START\|([^|]+)\|{re.escape(query_text[:50])}"
            query_match = re.search(query_pattern, log_content)
            
            if not query_match:
                return {"error": "Query not found in logs"}
            
            query_id = query_match.group(1)
            logger.debug(f"Found query ID: {query_id}")
            
            # Extract all log entries for this query ID
            rag_info = {
                "query_id": query_id,
                "retrieval_info": {
                    "chunks_retrieved": [],
                    "chunks_used_in_context": [],
                    "relevance_scores": [],
                    "source_files": [],
                    "retrieval_threshold": None,
                    "prompt_type": None
                },
                "generation_info": {
                    "prompt_built": None,
                    "context_length": 0,
                    "model_used": "qwen2.5:8b",
                    "generation_params": {}
                },
                "performance_metrics": {
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "total_processing_time": 0
                }
            }
            
            # Parse retrieval information
            retrieval_pattern = rf"RAG_RETRIEVAL\|{query_id}\|retrieved_count:(\d+)\|time:([\d.]+)s"
            retrieval_match = re.search(retrieval_pattern, log_content)
            if retrieval_match:
                rag_info["performance_metrics"]["retrieval_time"] = float(retrieval_match.group(2))
            
            # Parse chunk information
            chunk_pattern = rf"RAG_CHUNK\|{query_id}\|chunk_(\d+)\|file:([^|]+)\|score:([\d.]+)\|content_preview:([^|]+)"
            chunk_matches = re.findall(chunk_pattern, log_content)
            for chunk_idx, file_name, score, content_preview in chunk_matches:
                rag_info["retrieval_info"]["chunks_retrieved"].append({
                    "chunk_index": int(chunk_idx),
                    "source_file": file_name,
                    "relevance_score": float(score),
                    "content_preview": content_preview
                })
                rag_info["retrieval_info"]["relevance_scores"].append(float(score))
                if file_name not in rag_info["retrieval_info"]["source_files"]:
                    rag_info["retrieval_info"]["source_files"].append(file_name)
            
            # Parse context chunks (actually used)
            context_pattern = rf"RAG_CONTEXT_CHUNK\|{query_id}\|selected_(\d+)\|file:([^|]+)\|chunk_id:(\d+)\|score:([\d.]+)"
            context_matches = re.findall(context_pattern, log_content)
            for selected_idx, file_name, chunk_id, score in context_matches:
                rag_info["retrieval_info"]["chunks_used_in_context"].append({
                    "selected_index": int(selected_idx),
                    "source_file": file_name,
                    "chunk_id": int(chunk_id),
                    "relevance_score": float(score)
                })
            
            # Parse prompt type
            prompt_type_pattern = rf"RAG_PROMPT_TYPE\|{query_id}\|([^|]+)\|"
            prompt_type_match = re.search(prompt_type_pattern, log_content)
            if prompt_type_match:
                rag_info["retrieval_info"]["prompt_type"] = prompt_type_match.group(1)
            
            # Parse threshold analysis
            threshold_pattern = rf"RAG_THRESHOLD_ANALYSIS\|{query_id}\|.*dynamic_threshold:([\d.]+)"
            threshold_match = re.search(threshold_pattern, log_content)
            if threshold_match:
                rag_info["retrieval_info"]["retrieval_threshold"] = float(threshold_match.group(1))
            
            # Parse prompt building
            prompt_built_pattern = rf"RAG_PROMPT_BUILT\|{query_id}\|length:(\d+)\|context_docs:(\d+)"
            prompt_built_match = re.search(prompt_built_pattern, log_content)
            if prompt_built_match:
                rag_info["generation_info"]["context_length"] = int(prompt_built_match.group(1))
            
            # Parse full prompt
            full_prompt_pattern = rf"RAG_FULL_PROMPT\|{query_id}\|(.*?)(?=\n\d{{4}}-\d{{2}}-\d{{2}}|\n🔧 RAG_|\nRAG_|$)"
            full_prompt_match = re.search(full_prompt_pattern, log_content, re.DOTALL)
            if full_prompt_match:
                rag_info["generation_info"]["prompt_built"] = full_prompt_match.group(1).strip()
            
            # Parse generation time
            generation_pattern = rf"RAG_GENERATION\|{query_id}\|time:([\d.]+)s\|response_length:(\d+)"
            generation_match = re.search(generation_pattern, log_content)
            if generation_match:
                rag_info["performance_metrics"]["generation_time"] = float(generation_match.group(1))
            
            # Parse total time
            end_pattern = rf"RAG_QUERY_END\|{query_id}\|.*total_time:([\d.]+)s"
            end_match = re.search(end_pattern, log_content)
            if end_match:
                rag_info["performance_metrics"]["total_processing_time"] = float(end_match.group(1))
            
            return rag_info
            
        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
            return {"error": f"Log parsing failed: {str(e)}"}
    
    
    async def check_health(self) -> Tuple[bool, TimingMetrics]:
        """Check if the API is healthy and ready with detailed timing"""
        timing = TimingMetrics()
        start_time = time.time()
        
        try:
            health_start = time.time()
            async with self.session.get(f"{self.base_url}/health") as response:
                health_end = time.time()
                timing.health_check_time = health_end - health_start
                timing.network_request_time = timing.health_check_time
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"API Health Check: {data} (took {timing.health_check_time:.3f}s)")
                    timing.total_operation_time = time.time() - start_time
                    return True, timing
                else:
                    logger.error(f"Health check failed: {response.status}")
                    timing.total_operation_time = time.time() - start_time
                    return False, timing
        except Exception as e:
            logger.error(f"Health check error: {e}")
            timing.total_operation_time = time.time() - start_time
            return False, timing
    
    async def send_query(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """Send a query and capture detailed response information with comprehensive timing"""
        timing = TimingMetrics()
        overall_start = time.time()
        
        # Step 1: Prepare request
        prep_start = time.time()
        headers = {
            "Content-Type": "application/json",
            "X-Company-Pin": TEST_COMPANY_PIN,
            "X-API-Key": TEST_API_KEY,
            "X-User-Name": TEST_USER_NAME,
            "X-Client-ID": TEST_USER_ID
        }
        
        payload = {
            "query": question,
            "max_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.7
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        prep_end = time.time()
        timing.request_preparation_time = prep_end - prep_start
        
        query_start_datetime = datetime.now()
        
        try:
            # Step 2: Send network request
            network_start = time.time()
            async with self.session.post(
                f"{self.base_url}/query",
                headers=headers,
                json=payload
            ) as response:
                network_end = time.time()
                timing.network_request_time = network_end - network_start
                
                # Step 3: Process response
                response_proc_start = time.time()
                if response.status == 200:
                    data = await response.json()
                    response_proc_end = time.time()
                    timing.response_processing_time = response_proc_end - response_proc_start
                    
                    # Extract detailed information
                    result = {
                        "success": True,
                        "question": question,
                        "response": data.get("response", ""),
                        "session_id": data.get("session_id", ""),
                        "response_time": timing.network_request_time,
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status,
                        "question_length": len(question),
                        "response_length": len(data.get("response", "")),
                        "tokens_estimated": self.estimate_tokens(question + data.get("response", "")),
                        "timing_metrics": timing.to_dict()
                    }
                    
                    # Step 4: Parse RAG details from logs with timing
                    log_parse_start = time.time()
                    await asyncio.sleep(1)  # Give logs time to be written
                    rag_details = self.parse_rag_logs_for_query(question, query_start_datetime)
                    log_parse_end = time.time()
                    timing.log_parsing_time = log_parse_end - log_parse_start
                    
                    # Extract RAG timing from logs
                    if not rag_details.get("error"):
                        perf_metrics = rag_details.get("performance_metrics", {})
                        timing.rag_retrieval_time = perf_metrics.get("retrieval_time", 0.0)
                        timing.rag_generation_time = perf_metrics.get("generation_time", 0.0)
                        timing.rag_total_processing_time = perf_metrics.get("total_processing_time", 0.0)
                        
                        # Estimate database operation times from RAG processing
                        timing.db_session_lookup_time = max(0.001, timing.rag_total_processing_time * 0.05)  # ~5% for session lookup
                        timing.db_session_save_time = max(0.001, timing.rag_total_processing_time * 0.03)   # ~3% for session save
                    
                    timing.total_operation_time = time.time() - overall_start
                    result["timing_metrics"] = timing.to_dict()
                    result["rag_details"] = rag_details
                    
                    # Log detailed timing breakdown
                    logger.info(f"🕐 Timing Breakdown for query:")
                    logger.info(f"   Request prep: {timing.request_preparation_time:.3f}s")
                    logger.info(f"   Network request: {timing.network_request_time:.3f}s")
                    logger.info(f"   Response processing: {timing.response_processing_time:.3f}s")
                    logger.info(f"   Log parsing: {timing.log_parsing_time:.3f}s")
                    logger.info(f"   RAG retrieval: {timing.rag_retrieval_time:.3f}s")
                    logger.info(f"   RAG generation: {timing.rag_generation_time:.3f}s")
                    logger.info(f"   DB session ops: {timing.db_session_lookup_time + timing.db_session_save_time:.3f}s")
                    logger.info(f"   Total operation: {timing.total_operation_time:.3f}s")
                    
                    return result
                else:
                    error_text = await response.text()
                    response_proc_end = time.time()
                    timing.response_processing_time = response_proc_end - response_proc_start
                    timing.total_operation_time = time.time() - overall_start
                    
                    return {
                        "success": False,
                        "question": question,
                        "error": error_text,
                        "response_time": timing.network_request_time,
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status,
                        "timing_metrics": timing.to_dict()
                    }
                    
        except Exception as e:
            timing.total_operation_time = time.time() - overall_start
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "response_time": timing.total_operation_time,
                "timestamp": datetime.now().isoformat(),
                "status_code": None,
                "timing_metrics": timing.to_dict()
            }
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    async def run_sequential_test(self) -> List[Dict[str, Any]]:
        """Run all HR questions sequentially with detailed timing"""
        logger.info(f"🎯 Starting sequential HR questions test with {len(HR_QUESTIONS)} questions")
        
        # Check API health first with timing
        health_check_start = time.time()
        health_ok, health_timing = await self.check_health()
        health_check_end = time.time()
        
        if not health_ok:
            raise Exception("API health check failed - cannot proceed with tests")
        
        logger.info(f"✅ API health check completed in {health_check_end - health_check_start:.3f}s")
        
        results = []
        session_id = None
        total_test_start = time.time()
        
        # Track cumulative timing statistics
        cumulative_timings = {
            "total_request_prep_time": 0.0,
            "total_network_time": 0.0,
            "total_response_proc_time": 0.0,
            "total_log_parsing_time": 0.0,
            "total_rag_retrieval_time": 0.0,
            "total_rag_generation_time": 0.0,
            "total_db_operations_time": 0.0,
            "total_operation_time": 0.0
        }
        
        for i, question in enumerate(HR_QUESTIONS, 1):
            logger.info(f"📝 Question {i}/{len(HR_QUESTIONS)}: {question[:100]}...")
            
            # Add small delay between questions to avoid overwhelming the system
            if i > 1:
                delay_start = time.time()
                await asyncio.sleep(2)
                delay_end = time.time()
                logger.debug(f"   Inter-question delay: {delay_end - delay_start:.3f}s")
            
            question_start = time.time()
            result = await self.send_query(question, session_id)
            question_end = time.time()
            
            # Add question-level timing
            result["question_total_time"] = question_end - question_start
            result["question_number"] = i
            
            results.append(result)
            
            # Update cumulative timings
            if result.get("timing_metrics"):
                tm = result["timing_metrics"]
                cumulative_timings["total_request_prep_time"] += tm.get("request_preparation_time", 0)
                cumulative_timings["total_network_time"] += tm.get("network_request_time", 0)
                cumulative_timings["total_response_proc_time"] += tm.get("response_processing_time", 0)
                cumulative_timings["total_log_parsing_time"] += tm.get("log_parsing_time", 0)
                cumulative_timings["total_rag_retrieval_time"] += tm.get("rag_retrieval_time", 0)
                cumulative_timings["total_rag_generation_time"] += tm.get("rag_generation_time", 0)
                cumulative_timings["total_db_operations_time"] += tm.get("db_session_lookup_time", 0) + tm.get("db_session_save_time", 0)
                cumulative_timings["total_operation_time"] += tm.get("total_operation_time", 0)
            
            # Save incremental results after each question
            self.save_incremental_results(results, TEST_RESULTS_FILE.replace('.json', '_incremental.json'), cumulative_timings)
            
            # Update session_id for conversation continuity
            if result["success"] and result.get("session_id"):
                session_id = result["session_id"]
            
            if result["success"]:
                logger.info(f"✅ Question {i} completed in {result['response_time']:.2f}s (total: {result['question_total_time']:.2f}s)")
                logger.info(f"   Response length: {result['response_length']} chars")
                
                # Log RAG details if available
                rag_details = result.get("rag_details", {})
                if not rag_details.get("error"):
                    retrieval_info = rag_details.get("retrieval_info", {})
                    chunks_used = len(retrieval_info.get("chunks_used_in_context", []))
                    prompt_type = retrieval_info.get("prompt_type", "Unknown")
                    logger.info(f"   Chunks used: {chunks_used}")
                    logger.info(f"   Prompt type: {prompt_type}")
                    
                    # Log performance metrics
                    perf_metrics = rag_details.get("performance_metrics", {})
                    if perf_metrics.get("retrieval_time"):
                        logger.info(f"   Retrieval time: {perf_metrics['retrieval_time']:.3f}s")
                    if perf_metrics.get("generation_time"):
                        logger.info(f"   Generation time: {perf_metrics['generation_time']:.3f}s")
            else:
                logger.error(f"❌ Question {i} failed: {result.get('error', 'Unknown error')}")
        
        total_test_end = time.time()
        total_test_time = total_test_end - total_test_start
        
        # Add overall test timing summary
        logger.info(f"🏁 Test completed in {total_test_time:.2f}s")
        logger.info(f"📊 Cumulative timing breakdown:")
        logger.info(f"   Total request preparation: {cumulative_timings['total_request_prep_time']:.3f}s")
        logger.info(f"   Total network time: {cumulative_timings['total_network_time']:.3f}s")
        logger.info(f"   Total response processing: {cumulative_timings['total_response_proc_time']:.3f}s")
        logger.info(f"   Total RAG retrieval: {cumulative_timings['total_rag_retrieval_time']:.3f}s")
        logger.info(f"   Total RAG generation: {cumulative_timings['total_rag_generation_time']:.3f}s")
        logger.info(f"   Total DB operations: {cumulative_timings['total_db_operations_time']:.3f}s")
        logger.info(f"   Total operation time: {cumulative_timings['total_operation_time']:.3f}s")
        
        return results
    
    def save_incremental_results(self, results: List[Dict[str, Any]], filename: str, cumulative_timings: Dict[str, float] = None):
        """Save results incrementally after each question with timing analysis"""
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        # Calculate current statistics
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            response_lengths = [r["response_length"] for r in successful_results]
            question_lengths = [r["question_length"] for r in successful_results]
            
            # Calculate timing statistics
            timing_stats = {}
            if cumulative_timings:
                timing_stats = {
                    "cumulative_timings": cumulative_timings,
                    "average_timings": {
                        "avg_request_prep_time": cumulative_timings["total_request_prep_time"] / len(successful_results),
                        "avg_network_time": cumulative_timings["total_network_time"] / len(successful_results),
                        "avg_response_proc_time": cumulative_timings["total_response_proc_time"] / len(successful_results),
                        "avg_rag_retrieval_time": cumulative_timings["total_rag_retrieval_time"] / len(successful_results),
                        "avg_rag_generation_time": cumulative_timings["total_rag_generation_time"] / len(successful_results),
                        "avg_db_operations_time": cumulative_timings["total_db_operations_time"] / len(successful_results),
                        "avg_total_operation_time": cumulative_timings["total_operation_time"] / len(successful_results)
                    }
                }
            
            stats = {
                "total_questions": len(results),
                "successful_questions": len(successful_results),
                "failed_questions": len(failed_results),
                "success_rate": (len(successful_results) / len(results) * 100) if results else 0,
                "response_time_stats": {
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "min": min(response_times),
                    "max": max(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                "response_length_stats": {
                    "mean": statistics.mean(response_lengths),
                    "median": statistics.median(response_lengths),
                    "min": min(response_lengths),
                    "max": max(response_lengths)
                },
                "question_length_stats": {
                    "mean": statistics.mean(question_lengths),
                    "median": statistics.median(question_lengths),
                    "min": min(question_lengths),
                    "max": max(question_lengths)
                },
                "timing_analysis": timing_stats
            }
        else:
            stats = {
                "total_questions": len(results),
                "successful_questions": 0,
                "failed_questions": len(failed_results),
                "success_rate": 0,
                "response_time_stats": {},
                "response_length_stats": {},
                "question_length_stats": {},
                "timing_analysis": {}
            }
        
        # Create incremental report
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "Sequential HR Questions Test (In Progress)",
                "progress": f"{len(results)}/{len(HR_QUESTIONS)} questions completed",
                "user_info": {
                    "user_name": TEST_USER_NAME,
                    "user_id": TEST_USER_ID,
                    "company_pin": TEST_COMPANY_PIN
                },
                "current_statistics": stats
            },
            "detailed_results": results,
            "latest_result": results[-1] if results else None
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Incremental results saved: {len(results)}/{len(HR_QUESTIONS)} completed")
        
        # Log timing summary for current progress
        if cumulative_timings and len(successful_results) > 0:
            logger.info(f"⏱️  Current timing averages:")
            logger.info(f"   Avg network time: {cumulative_timings['total_network_time'] / len(successful_results):.3f}s")
            logger.info(f"   Avg RAG retrieval: {cumulative_timings['total_rag_retrieval_time'] / len(successful_results):.3f}s")
            logger.info(f"   Avg RAG generation: {cumulative_timings['total_rag_generation_time'] / len(successful_results):.3f}s")
            logger.info(f"   Avg DB operations: {cumulative_timings['total_db_operations_time'] / len(successful_results):.3f}s")

    def analyze_and_save_results(self, results: List[Dict[str, Any]], filename: str):
        """Analyze results and save detailed report"""
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        # Calculate statistics
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            response_lengths = [r["response_length"] for r in successful_results]
            question_lengths = [r["question_length"] for r in successful_results]
            
            # Calculate detailed timing statistics
            timing_analysis = self.calculate_timing_statistics(successful_results)
            
            stats = {
                "total_questions": len(results),
                "successful_questions": len(successful_results),
                "failed_questions": len(failed_results),
                "success_rate": (len(successful_results) / len(results) * 100) if results else 0,
                "response_time_stats": {
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "min": min(response_times),
                    "max": max(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                "response_length_stats": {
                    "mean": statistics.mean(response_lengths),
                    "median": statistics.median(response_lengths),
                    "min": min(response_lengths),
                    "max": max(response_lengths)
                },
                "question_length_stats": {
                    "mean": statistics.mean(question_lengths),
                    "median": statistics.median(question_lengths),
                    "min": min(question_lengths),
                    "max": max(question_lengths)
                },
                "detailed_timing_analysis": timing_analysis
            }
        else:
            stats = {
                "total_questions": len(results),
                "successful_questions": 0,
                "failed_questions": len(failed_results),
                "success_rate": 0,
                "response_time_stats": {},
                "response_length_stats": {},
                "question_length_stats": {},
                "detailed_timing_analysis": {}
            }
        
        # Categorize questions by topic (updated categories)
        categories = {
            "Employee Exit & Separation": [r for r in results if any(keyword in r["question"].lower() 
                            for keyword in ["exit", "clearance", "separation", "settlement", "exit interview"])],
            "Payroll & Compensation": [r for r in results if any(keyword in r["question"].lower() 
                       for keyword in ["payroll", "salary", "allowance", "deduction", "overtime", "loan", "bonus", "pf", "provident"])],
            "Employee Management": [r for r in results if any(keyword in r["question"].lower() 
                                  for keyword in ["employee", "transfer", "transition", "resignation", "rehire", "onboarding"])],
            "Leave Management": [r for r in results if any(keyword in r["question"].lower() 
                               for keyword in ["leave", "vacation", "quota"])],
            "Attendance & Time": [r for r in results if any(keyword in r["question"].lower() 
                          for keyword in ["attendance", "shift", "remote work", "time"])],
            "HR Document Generation": [r for r in results if any(keyword in r["question"].lower() 
                                     for keyword in ["job description", "kpi", "policy", "checklist", "appraisal", "structure", "training plan", "questionnaire", "sop", "hierarchy"])],
            "System Configuration": [r for r in results if any(keyword in r["question"].lower() 
                                   for keyword in ["configuration", "setup", "template", "workflow", "approval", "bulk upload"])]
        }
        
        # Create comprehensive report
        report = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "Sequential HR Questions Test",
                "user_info": {
                    "user_name": TEST_USER_NAME,
                    "user_id": TEST_USER_ID,
                    "company_pin": TEST_COMPANY_PIN
                },
                "statistics": stats,
                "categories": {cat: len(questions) for cat, questions in categories.items()}
            },
            "detailed_results": results,
            "category_analysis": {},
            "timing_breakdown_analysis": self.analyze_timing_patterns(successful_results) if successful_results else {}
        }
        
        # Analyze each category
        for category, cat_results in categories.items():
            if cat_results:
                successful_cat = [r for r in cat_results if r.get("success")]
                if successful_cat:
                    cat_times = [r["response_time"] for r in successful_cat]
                    cat_timing_analysis = self.calculate_timing_statistics(successful_cat)
                    
                    report["category_analysis"][category] = {
                        "total_questions": len(cat_results),
                        "successful": len(successful_cat),
                        "success_rate": len(successful_cat) / len(cat_results) * 100,
                        "avg_response_time": statistics.mean(cat_times),
                        "min_response_time": min(cat_times),
                        "max_response_time": max(cat_times),
                        "timing_breakdown": cat_timing_analysis
                    }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Test results saved to {filename}")
        
        # Print summary
        self.print_summary(stats, categories, successful_results, failed_results)
    
    def print_summary(self, stats, categories, successful_results, failed_results):
        """Print detailed test summary with timing breakdown"""
        print("\n" + "="*80)
        print("🎯 HR QUESTIONS SEQUENTIAL TEST SUMMARY")
        print("="*80)
        print(f"📊 Total Questions: {stats['total_questions']}")
        print(f"✅ Successful: {stats['successful_questions']}")
        print(f"❌ Failed: {stats['failed_questions']}")
        print(f"📈 Success Rate: {stats['success_rate']:.1f}%")
        
        if stats.get('response_time_stats'):
            rt_stats = stats['response_time_stats']
            print(f"⏱️  Response Time Stats:")
            print(f"   Mean: {rt_stats['mean']:.2f}s")
            print(f"   Median: {rt_stats['median']:.2f}s")
            print(f"   Min: {rt_stats['min']:.2f}s")
            print(f"   Max: {rt_stats['max']:.2f}s")
            print(f"   Std Dev: {rt_stats['std_dev']:.2f}s")
        
        # Show detailed timing breakdown if available
        timing_analysis = stats.get('detailed_timing_analysis', {})
        if timing_analysis:
            print(f"\n🔧 DETAILED TIMING BREAKDOWN:")
            print("-" * 50)
            
            # Show average times for each component
            for component, component_stats in timing_analysis.items():
                if isinstance(component_stats, dict) and 'mean' in component_stats:
                    component_name = component.replace('_', ' ').title()
                    print(f"{component_name:25} | Avg: {component_stats['mean']:.3f}s | "
                          f"Min: {component_stats['min']:.3f}s | Max: {component_stats['max']:.3f}s")
        
        print("\n📋 CATEGORY BREAKDOWN:")
        print("-" * 50)
        for category, questions in categories.items():
            successful_cat = [q for q in questions if q.get("success")]
            success_rate = len(successful_cat) / len(questions) * 100 if questions else 0
            avg_time = statistics.mean([q["response_time"] for q in successful_cat]) if successful_cat else 0
            print(f"{category:20} | Total: {len(questions):2d} | Success: {len(successful_cat):2d} | "
                  f"Rate: {success_rate:5.1f}% | Avg Time: {avg_time:5.2f}s")
        
        # Show sample responses
        if successful_results:
            print(f"\n💬 SAMPLE SUCCESSFUL RESPONSES (First 3):")
            print("-" * 80)
            for i, result in enumerate(successful_results[:3], 1):
                print(f"\n{i}. Question: {result['question'][:100]}...")
                print(f"   Response: {result['response'][:200]}...")
                print(f"   Time: {result['response_time']:.2f}s | Length: {result['response_length']} chars")
                
                # Show timing breakdown for this result
                tm = result.get('timing_metrics', {})
                if tm:
                    print(f"   Timing: Network: {tm.get('network_request_time', 0):.3f}s | "
                          f"RAG: {tm.get('rag_retrieval_time', 0) + tm.get('rag_generation_time', 0):.3f}s | "
                          f"DB: {tm.get('db_session_lookup_time', 0) + tm.get('db_session_save_time', 0):.3f}s")
        
        # Show failed questions
        if failed_results:
            print(f"\n❌ FAILED QUESTIONS:")
            print("-" * 80)
            for result in failed_results:
                print(f"Question: {result['question'][:100]}...")
                print(f"Error: {result.get('error', 'Unknown error')}")
                print("-" * 40)
    
    def calculate_timing_statistics(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed timing statistics from successful results"""
        if not successful_results:
            return {}
        
        timing_metrics = []
        for result in successful_results:
            tm = result.get("timing_metrics", {})
            if tm:
                timing_metrics.append(tm)
        
        if not timing_metrics:
            return {}
        
        # Calculate statistics for each timing component
        components = [
            "request_preparation_time", "network_request_time", "response_processing_time",
            "log_parsing_time", "rag_retrieval_time", "rag_generation_time",
            "db_session_lookup_time", "db_session_save_time", "total_operation_time"
        ]
        
        timing_stats = {}
        for component in components:
            values = [tm.get(component, 0) for tm in timing_metrics if tm.get(component, 0) > 0]
            if values:
                timing_stats[component] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "total": sum(values),
                    "count": len(values)
                }
        
        return timing_stats
    
    def analyze_timing_patterns(self, successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing patterns and identify bottlenecks"""
        if not successful_results:
            return {}
        
        timing_analysis = {
            "bottleneck_analysis": {},
            "performance_trends": {},
            "efficiency_metrics": {}
        }
        
        # Identify bottlenecks by analyzing which component takes the most time on average
        timing_components = {}
        for result in successful_results:
            tm = result.get("timing_metrics", {})
            if tm:
                for component, value in tm.items():
                    if isinstance(value, (int, float)) and value > 0:
                        if component not in timing_components:
                            timing_components[component] = []
                        timing_components[component].append(value)
        
        # Calculate average time for each component
        avg_times = {}
        for component, values in timing_components.items():
            if values:
                avg_times[component] = statistics.mean(values)
        
        # Identify top bottlenecks
        if avg_times:
            sorted_components = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
            timing_analysis["bottleneck_analysis"] = {
                "primary_bottleneck": sorted_components[0] if sorted_components else None,
                "top_3_bottlenecks": sorted_components[:3],
                "component_percentages": {
                    comp: (time_val / sum(avg_times.values()) * 100) 
                    for comp, time_val in avg_times.items()
                }
            }
        
        # Analyze performance trends over time
        if len(successful_results) > 5:
            first_half = successful_results[:len(successful_results)//2]
            second_half = successful_results[len(successful_results)//2:]
            
            first_half_avg = statistics.mean([r["response_time"] for r in first_half])
            second_half_avg = statistics.mean([r["response_time"] for r in second_half])
            
            timing_analysis["performance_trends"] = {
                "first_half_avg_response_time": first_half_avg,
                "second_half_avg_response_time": second_half_avg,
                "performance_change": ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0,
                "trend": "improving" if second_half_avg < first_half_avg else "degrading"
            }
        
        return timing_analysis


async def main():
    """Main test execution function"""
    logger.info("🚀 Starting Sequential HR Questions Test for FlowHCM RAG Chatbot")
    
    async with HRQuestionsTester(API_BASE_URL) as tester:
        try:
            # Run the sequential test
            results = await tester.run_sequential_test()
            
            # Analyze and save results
            tester.analyze_and_save_results(results, TEST_RESULTS_FILE)
            
            logger.info("🎉 Sequential HR questions test completed successfully")
            logger.info("💾 All conversations and analysis saved to file")
            
        except Exception as e:
            logger.error(f"💥 Test execution failed: {e}")
            raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())