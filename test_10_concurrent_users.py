#!/usr/bin/env python3
"""
10 Concurrent Users Test Script for FlowHCM RAG Chatbot
Simulates 10 users asking questions simultaneously and stores conversations in database.
"""

import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test Configuration
API_BASE_URL = "http://localhost:8000"
TEST_RESULTS_FILE = "test_10_users_results.json"
CONCURRENT_USERS = 10

# Test credentials - matching your database
TEST_COMPANY_PIN = "11032"
TEST_API_KEY = "test123456789012345678901234567890"

# Expanded sample queries for testing
SAMPLE_QUERIES = [
    # Leave and Time Off
    "What is the company's leave policy?",
    "How do I apply for annual leave?",
    "How many sick days do I get per year?",
    "how do i change shift?",
    "how to make an attendance request/",
    "How do I request emergency leave?",
    "What is the bereavement leave policy?",
    "How do I apply for unpaid leave?",
    
    # Work Policies
    "What are the working hours?",
    "What is the remote work policy?",
    "What is the overtime policy?",
    "What is the dress code policy?",
    "What is the attendance policy?",
    "Can I work flexible hours?",
    "What is the work from home policy?",
    "What are the break time policies?",
    
    # Benefits and Compensation
    "Tell me about employee benefits",
    "How do I access my payslip?",
    "What health insurance benefits do we have?",
    "What is the retirement plan?",
    "Are there any wellness programs?",
    "What are the performance bonuses?",
    "Do we have life insurance coverage?",
    "What are the transportation allowances?",
    
    # HR Processes
    "What is the performance review process?",
    "How do I update my personal information?",
    "What is the grievance procedure?",
    "How do I report a workplace incident?",
    "What is the employee referral program?",
    "How do I request training?",
    "What is the promotion process?",
    "How do I change my department?",
    
    # Expenses and Finance
    "How do I submit an expense report?",
    "What expenses can I claim?",
    "What is the travel expense policy?",
    "How do I get reimbursed for business meals?",
    "What is the mobile phone allowance policy?",
    "How do I claim medical expenses?",
    
    # Company Information
    "What is the company's mission statement?",
    "What are the company values?",
    "What are the company holidays?",
    "How do I book a meeting room?",
    "What are the IT security policies?",
    "What is the code of conduct?",
    "What are the safety protocols?",
    "How do I access company resources?",
    
    # Training and Development
    "What training programs are available?",
    "How do I enroll in professional development courses?",
    "What is the reimbursement policy?",
    "Are there mentorship programs?",
    "What certifications does the company support?",
    "whats flowhcm?",
    
    # IT and Equipment
    "write me kpi for Account Executive",
    "What is the laptop policy?",
    "How do I reset my password?",
    "create job description for Product owner",
    "How do i change all employees shift?",
    "What is the BYOD policy?"
]


class ConcurrentUserTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        # Create session with connection limits
        connector = aiohttp.TCPConnector(
            limit=100,              # Total connection pool size
            limit_per_host=30,      # Max connections per host
            ttl_dns_cache=300,      # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,   # Keep connections alive
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> bool:
        """Check if the API is healthy and ready"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"API Health Check: {data}")
                    return True
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def send_query(self, user_id: str, user_name: str, message: str, session_id: str = None) -> Dict[str, Any]:
        """Send a query using the /query endpoint with authentication and detailed timing"""
        headers = {
            "Content-Type": "application/json",
            "X-Company-Pin": TEST_COMPANY_PIN,
            "X-API-Key": TEST_API_KEY,
            "X-User-Name": user_name,
            "X-Client-ID": user_id
        }
        
        payload = {
            "query": message,
            "max_tokens": 400,
            "temperature": 0.1,
            "top_p": 0.7
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        # Detailed client-side timing
        timing_log = {}
        overall_start = time.time()
        timing_log["request_start"] = overall_start
        
        try:
            # Connection establishment timing
            connection_start = time.time()
            
            async with self.session.post(
                f"{self.base_url}/query",
                headers=headers,
                json=payload
            ) as response:
                timing_log["connection_established"] = time.time()
                
                # Response reading timing
                response_read_start = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    timing_log["response_read_complete"] = time.time()
                    
                    # Calculate timing breakdown
                    total_time = timing_log["response_read_complete"] - timing_log["request_start"]
                    connection_time = timing_log["connection_established"] - timing_log["request_start"]
                    response_read_time = timing_log["response_read_complete"] - timing_log["connection_established"]
                    
                    return {
                        "success": True,
                        "user_id": user_id,
                        "user_name": user_name,
                        "query": message,
                        "response": data.get("response", ""),
                        "session_id": data.get("session_id", ""),
                        "response_time": total_time,
                        "timing_breakdown": {
                            "total_time": total_time,
                            "connection_time": connection_time,
                            "server_processing_time": total_time - connection_time - response_read_time,
                            "response_read_time": response_read_time
                        },
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    timing_log["error_read_complete"] = time.time()
                    total_time = timing_log["error_read_complete"] - timing_log["request_start"]
                    
                    return {
                        "success": False,
                        "user_id": user_id,
                        "user_name": user_name,
                        "query": message,
                        "error": error_text,
                        "response_time": total_time,
                        "timing_breakdown": {
                            "total_time": total_time,
                            "connection_time": timing_log.get("connection_established", time.time()) - timing_log["request_start"],
                            "error_processing_time": total_time - (timing_log.get("connection_established", time.time()) - timing_log["request_start"])
                        },
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status
                    }
                    
        except Exception as e:
            error_time = time.time()
            total_time = error_time - overall_start
            
            return {
                "success": False,
                "user_id": user_id,
                "user_name": user_name,
                "query": message,
                "error": str(e),
                "response_time": total_time,
                "timing_breakdown": {
                    "total_time": total_time,
                    "error_type": "connection_error"
                },
                "timestamp": datetime.now().isoformat(),
                "status_code": None
            }
    
    async def simulate_user(self, user_id: str, user_name: str, num_queries: int = 6) -> List[Dict[str, Any]]:
        """Simulate a user sending multiple queries with session continuity"""
        logger.info(f"🚀 Starting simulation for {user_name} ({user_id})")
        
        user_results = []
        user_queries = random.sample(SAMPLE_QUERIES, min(num_queries, len(SAMPLE_QUERIES)))
        session_id = None  # Will be set after first query
        
        for i, query in enumerate(user_queries):
            # Add random delay between queries (1-8 seconds)
            if i > 0:
                delay = random.uniform(1, 8)
                await asyncio.sleep(delay)
            
            logger.info(f"💬 {user_name} asking: {query}")
            
            result = await self.send_query(user_id, user_name, query, session_id)
            user_results.append(result)
            
            # Update session_id for subsequent queries to maintain conversation context
            if result["success"] and result.get("session_id"):
                session_id = result["session_id"]
            
            if result["success"]:
                logger.info(f"✅ {user_name} got response in {result['response_time']:.2f}s")
            else:
                logger.error(f"❌ {user_name} query failed: {result.get('error', 'Unknown error')}")
        
        logger.info(f"🏁 Completed simulation for {user_name}")
        return user_results
    
    async def run_concurrent_test(self, num_users: int, queries_per_user: int = 6) -> List[Dict[str, Any]]:
        """Run concurrent user simulations"""
        logger.info(f"🎯 Starting concurrent test with {num_users} users, {queries_per_user} queries each")
        
        # Check API health first
        if not await self.check_health():
            raise Exception("API health check failed - cannot proceed with tests")
        
        # Generate user profiles
        user_names = [
            "Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Emma Brown",
            "Frank Miller", "Grace Lee", "Henry Taylor", "Ivy Chen", "Jack Anderson",
            "Kate Williams", "Liam Garcia", "Maya Patel", "Noah Rodriguez", "Olivia Martinez"
        ]
        
        users = []
        for i in range(num_users):
            user_id = f"test_user_{i+1:03d}"
            user_name = user_names[i % len(user_names)]
            if i >= len(user_names):
                user_name = f"{user_name}_{i//len(user_names)+1}"
            users.append((user_id, user_name))
        
        # Create user tasks
        tasks = [self.simulate_user(user_id, user_name, queries_per_user) for user_id, user_name in users]
        
        # Run all users concurrently
        logger.info("🔥 Launching all user simulations concurrently...")
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        flattened_results = []
        for i, user_results in enumerate(all_results):
            if isinstance(user_results, Exception):
                logger.error(f"❌ User {users[i][1]} simulation failed: {user_results}")
                flattened_results.append({
                    "success": False,
                    "user_id": users[i][0],
                    "user_name": users[i][1],
                    "error": str(user_results),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                flattened_results.extend(user_results)
        
        return flattened_results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save test results to JSON file with detailed analysis"""
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        # Calculate statistics
        response_times = [r.get("response_time", 0) for r in successful_results if r.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # User statistics
        user_stats = {}
        for result in results:
            user_name = result.get("user_name", "Unknown")
            if user_name not in user_stats:
                user_stats[user_name] = {"total": 0, "successful": 0, "failed": 0, "avg_time": 0}
            
            user_stats[user_name]["total"] += 1
            if result.get("success"):
                user_stats[user_name]["successful"] += 1
                if result.get("response_time"):
                    user_stats[user_name]["avg_time"] += result["response_time"]
            else:
                user_stats[user_name]["failed"] += 1
        
        # Calculate average times per user
        for user_name in user_stats:
            if user_stats[user_name]["successful"] > 0:
                user_stats[user_name]["avg_time"] /= user_stats[user_name]["successful"]
        
        test_summary = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_users": len(set(r.get("user_name") for r in results if r.get("user_name"))),
                "total_queries": len(results),
                "successful_queries": len(successful_results),
                "failed_queries": len(failed_results),
                "success_rate": (len(successful_results) / len(results) * 100) if results else 0,
                "average_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "user_statistics": user_stats
            },
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Test results saved to {filename}")
        
        # Print detailed summary
        print("\n" + "="*80)
        print("🎯 10 CONCURRENT USERS TEST SUMMARY")
        print("="*80)
        print(f"📊 Total Users: {test_summary['test_info']['total_users']}")
        print(f"📊 Total Queries: {test_summary['test_info']['total_queries']}")
        print(f"✅ Successful Queries: {test_summary['test_info']['successful_queries']}")
        print(f"❌ Failed Queries: {test_summary['test_info']['failed_queries']}")
        print(f"📈 Success Rate: {test_summary['test_info']['success_rate']:.1f}%")
        print(f"⏱️  Average Response Time: {test_summary['test_info']['average_response_time']:.2f}s")
        print(f"⚡ Min Response Time: {test_summary['test_info']['min_response_time']:.2f}s")
        print(f"🐌 Max Response Time: {test_summary['test_info']['max_response_time']:.2f}s")
        print("="*80)
        
        # User performance breakdown
        print("\n👥 USER PERFORMANCE BREAKDOWN:")
        print("-" * 80)
        for user_name, stats in user_stats.items():
            success_rate = (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"{user_name:20} | Total: {stats['total']:2d} | Success: {stats['successful']:2d} | "
                  f"Failed: {stats['failed']:2d} | Rate: {success_rate:5.1f}% | Avg Time: {stats['avg_time']:5.2f}s")
        
        # Show sample successful responses
        if successful_results:
            print("\n💬 SAMPLE SUCCESSFUL RESPONSES WITH TIMING:")
            print("-" * 80)
            for result in successful_results[:5]:  # Show first 5 successful responses
                print(f"👤 User: {result['user_name']} ({result['user_id']})")
                print(f"❓ Query: {result['query']}")
                print(f"🤖 Response: {result['response'][:150]}...")
                print(f"🆔 Session ID: {result.get('session_id', 'N/A')}")
                print(f"⏱️  Total Time: {result['response_time']:.2f}s")
                
                # Show timing breakdown if available
                if 'timing_breakdown' in result:
                    timing = result['timing_breakdown']
                    print(f"📊 Timing Breakdown:")
                    print(f"   ├── Connection: {timing.get('connection_time', 0):.3f}s")
                    print(f"   ├── Server Processing: {timing.get('server_processing_time', 0):.3f}s")
                    print(f"   └── Response Reading: {timing.get('response_read_time', 0):.3f}s")
                
                print("-" * 40)
        
        # Show failed queries if any
        if failed_results:
            print("\n❌ FAILED QUERIES:")
            print("-" * 80)
            for result in failed_results[:5]:  # Show first 5 failed queries
                print(f"👤 User: {result.get('user_name', 'N/A')} ({result['user_id']})")
                print(f"❓ Query: {result.get('query', 'N/A')}")
                print(f"💥 Error: {result.get('error', 'Unknown error')}")
                print("-" * 40)


async def main():
    """Main test execution function"""
    logger.info("🚀 Starting 10 concurrent users test for FlowHCM RAG Chatbot")
    
    async with ConcurrentUserTester(API_BASE_URL) as tester:
        try:
            # Run the concurrent test
            results = await tester.run_concurrent_test(CONCURRENT_USERS, queries_per_user=6)
            
            # Save results to file
            tester.save_results(results, TEST_RESULTS_FILE)
            
            logger.info("🎉 10 concurrent users test completed successfully")
            logger.info("💾 All conversations have been stored in the PostgreSQL database")
            
        except Exception as e:
            logger.error(f"💥 Test execution failed: {e}")
            raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())