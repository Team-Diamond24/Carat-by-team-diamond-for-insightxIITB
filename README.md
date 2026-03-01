# Carat - AI-Powered UPI Transaction Analytics Dashboard

**Carat** is an intelligent conversational analytics platform that allows users to query UPI transaction data using natural language. Built for Techfest IIT Bombay x NPCI by Team Diamond.

## Features

- **Natural Language Queries**: Ask questions in plain English about transaction data
- **AI-Powered SQL Generation**: Automatically converts questions to optimized SQL queries
- **Interactive Visualizations**: Dynamic charts with Plotly (bar, pie, line, scatter)
- **Comprehensive Analysis**: LLM-generated insights with key statistics
- **Real-time Dashboard**: Modern dark-themed UI with live KPIs
- **250K+ Transactions**: Full UPI transaction dataset for 2024

## Tech Stack

- **Backend**: Python, Flask, SQLite
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript, Plotly.js
- **AI**: OpenAI API (via OpenRouter) for SQL generation and analysis
- **Data**: 250,000 UPI transaction records

## Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API Key (or OpenAI API Key)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Carat-by-team-diamond-for-insightxIITB
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```env
OPENROUTER_API_KEY=your_api_key_here
PRIMARY_MODEL=openai/gpt-4o-mini
FALLBACK_MODEL=openai/gpt-3.5-turbo
SQLITE_DB_PATH=carat.db
```

4. Migrate data to SQLite:
```bash
python migrate_to_sqlite.py
```

5. Run the application:
```bash
python server.py
```

6. Open your browser:
- Landing Page: http://localhost:5050
- Dashboard: http://localhost:5050/dashboard

## Project Structure

```
Carat-by-team-diamond-for-insightxIITB/
├── server.py              # Flask server (port 5050)
├── sql_analyst.py         # SQL generation and analysis
├── llm_client.py          # OpenAI/OpenRouter API wrapper
├── db.py                  # SQLite database connection
├── shared_utils.py        # Shared utilities and caching
├── failure_logger.py      # Error logging system
├── dashboard.html         # Main analytics dashboard
├── index.html             # Landing page
├── migrate_to_sqlite.py   # Data migration script
├── requirements.txt       # Python dependencies
├── carat.db              # SQLite database (generated)
└── upi_transactions_2024.csv  # Source data
```

## Usage Examples

Ask questions like:
- "Which bank has the most failed transactions?"
- "What is the fraud rate by state?"
- "Show me transaction volume by device type"
- "Compare success rates between iOS and Android"
- "What are the peak transaction hours?"

## Data Schema

The `upi_transactions_2024` table contains:
- Transaction details (ID, type, amount, status)
- User demographics (age group, state)
- Banking info (sender/receiver banks)
- Device & network data (device type, network type)
- Fraud indicators
- Temporal data (hour, day, weekend flag)
- Merchant categories

## Features in Detail

### AI-Powered Analysis
- Automatic SQL query generation from natural language
- Intelligent chart type selection (bar, pie, line, scatter)
- Comprehensive 4-6 sentence analyst-grade summaries
- Follow-up question suggestions

### Visualizations
- Properly sorted charts (highest to lowest)
- Value labels on all bars
- Percentage formatting for rates
- Responsive design with consistent styling
- Color-coded categories

### Performance
- Query result caching (LRU cache, 200 entries)
- Fast-path queries for common questions
- Automatic fallback to secondary AI model
- SQL query validation and sanitization

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `PRIMARY_MODEL`: Primary AI model (default: openai/gpt-4o-mini)
- `FALLBACK_MODEL`: Fallback AI model (default: openai/gpt-3.5-turbo)
- `SQLITE_DB_PATH`: Database file path (default: carat.db)
- `RATE_LIMIT_PER_MINUTE`: API rate limit (default: 20)
- `ADMIN_KEY`: Admin endpoint access key (optional)

## Security

- SQL injection protection (whitelist SELECT queries only)
- Rate limiting on API endpoints
- Admin endpoints require authentication
- Input validation and sanitization

## Troubleshooting

### Port Issues
The app runs on port 5050 to avoid conflicts with common Windows services.

### API Errors
- **402 Error**: Check your OpenRouter credits
- **404 Error**: Verify model names in `.env`
- **429 Error**: Rate limit exceeded, wait or upgrade plan

### Database Issues
If the database is empty, run:
```bash
python migrate_to_sqlite.py
```

## Development

### Running Tests
```bash
cd tests
python -m pytest
```

### Code Structure
- `sql_analyst.py`: Handles SQL generation, execution, and chart creation
- `llm_client.py`: Manages AI model calls with retry logic
- `shared_utils.py`: Caching, validation, and utility functions

## License

Built for Techfest IIT Bombay x NPCI by Team Diamond.

## Support

For issues or questions, please open an issue in the repository.
