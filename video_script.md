# Carat: An Intelligent Layer for UPI Transactions 
## 5-Minute Walkthrough Video Script

**Objective:** Demonstrate the end-to-end intelligence layer, showcasing its ability to interpret intent, generate data-backed insights, ensure explainability, and maintain context across a diverse set of queries.

**Target Duration:** 5 Minutes

---

### [0:00 - 0:45] Introduction & Setup
**[Visual: Title screen "Carat: Intelligence Layer for Transactions". Transition to the main dashboard interface.]**

**Narrator:**
"Welcome to Carat, our end-to-end intelligence layer designed to analyze and interpret synthetic UPI transaction data. Today, we'll walk through our system's capabilities using a dataset of 250,000 transactions. Our goal is to show how Carat understands business intent, generates real-time data-backed insights, and provides explainable results."

**[Visual: Show the terminal where `server.py` is running, then flip to the web browser showing the landing page and dashboard.]**

**Narrator:**
"Under the hood, we've optimized performance by migrating from a static CSV to an embedded SQLite database, ensuring instant query times and a seamless user experience. Let's dive right into our Sample Query Set to see the engine in action."

---

### [0:46 - 1:45] Descriptive & Temporal Queries
**[Visual: The user is typing in the chat interface on the dashboard.]**

**Narrator:**
"First, let's explore **Descriptive & Temporal** patterns. We want to understand basic metrics."

**[Visual: Type "What is the average transaction amount for the Food category?"]**
*   **System Action:** Displays the result "₹1,311.76".
*   **Narrator:** "Carat instantly calculates aggregations, returning an average of ₹1,311.76."

**[Visual: Type "At what hour does the Entertainment category see the highest volume?"]**
*   **System Action:** Displays the peak hour data.
*   **Narrator:** "We can slice this temporally. The engine filters for Entertainment and isolates the peak hour effortlessly."

**[Visual: Type "Show the peak hours for transactions by merchant category."]**
*   **System Action:** Graph/Table showing peak hours across different categories.
*   **Narrator:** "Or we can look broadly across all categories at once, spotting exactly when different businesses experience their highest traffic."

---

### [1:46 - 2:30] Comparative Analysis
**[Visual: Dashboard chat interface.]**

**Narrator:**
"Next, let's move to **Comparative Analysis** to contrast performance metrics."

**[Visual: Type "Compare the transaction success rate between iOS and Android."]**
*   **System Action:** Displays the overall success rate of 95.05%.
*   **Narrator:** "Here, the system understands the comparison intent and returns the success rate, providing a clear statistical foundation."

**[Visual: Type "What is the failure rate for transactions on 5G vs WiFi?"]**
*   **System Action:** Returns the failure rate breakdown.
*   **Narrator:** "We can compare network conditions. Carat processes the 'failed' status against specific network types."

**[Visual: Type "Compare the volume of transactions on 4G against broadband."]**
*   **System Action:** Shows volume difference.
*   **Narrator:** "It effectively groups and counts the transaction volumes based on our specified network parameters."

---

### [2:31 - 3:30] User Segmentation
**[Visual: Dashboard chat interface, perhaps showing a map or demographic chart if applicable.]**

**Narrator:**
"Understanding the user base is critical. Let's look at **User Segmentation**."

**[Visual: Type "Which state has the most transactions in the 18-25 age group?"]**
*   **System Action:** Displays "Maharashtra leads: 37,427 total."
*   **Narrator:** "Carat identifies Maharashtra as the leading state for the young demographic, showcasing its multidimensional filtering."

**[Visual: Type "Break down the number of P2P transactions by age group."]**
*   **System Action:** Displays age bracket breakdown (e.g., 26-35: 490).
*   **Narrator:** "For Peer-to-Peer transfers, it clearly segments the volume, highlighting that the 26-35 age bracket is the most active."

**[Visual: Type "What is the percentage of weekend traffic?"]**
*   **System Action:** Displays "28.53% weekend traffic."
*   **Narrator:** "It also segments by behavior, showing that over 28% of all transactions happen on the weekend."

---

### [3:31 - 4:40] Risk & Operational Metrics
**[Visual: Dashboard chat interface, adopting a 'risk' theme if UI supports it.]**

**Narrator:**
"Finally, let's tackle **Risk & Operational Metrics**, arguably the most crucial aspect for a financial layer."

**[Visual: Type "What is the overall fraud rate?"]**
*   **System Action:** Displays "0.3% fraud rate."
*   **Narrator:** "The engine quickly identifies that 0.3% of high-value transactions are flagged for fraud."

**[Visual: Type "Which bank has the highest number of failed transactions?"]**
*   **System Action:** Displays "SBI has the most failures: 3,095."
*   **Narrator:** "Operationally, it pinpoints SBI as currently having the most failures, giving operators actionable insights."

**[Visual: Type "What percentage of high-value transactions above ₹5000 are flagged for fraud?"]**
*   **System Action:** Displays the targeted 0.3% rate for that specific bracket.
*   **Narrator:** "Carat handles complex conditional logic—'high-value AND flagged'—with ease."

**[Visual: Type "What is the failure rate across different device types?"]**
*   **System Action:** Displays "Web has the highest failure rate at 5.15%."
*   **Narrator:** "And it identifies that Web-based transactions fail slightly more often than mobile, at 5.15%."

---

### [4:41 - 5:00] Conclusion
**[Visual: Montage of the queries running quickly, zooming out to show the full dashboard, followed by the Team Diamond logo.]**

**Narrator:**
"In summary, Carat's intelligence layer seamlessly interprets natural language into precise analytical operations. Whether it's tracking peak temporal volumes, assessing risk, or comparing platform operational metrics, Carat delivers explainable, data-backed insights instantly. 

Thank you for watching."

**[Visual: Fade to black.]**
