import os
import pandas as pd
import requests
from typing import Optional, List
import time
from datetime import datetime

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices

def safe_float(value, default=0.0):
    """Convert value to float safely, returning default if conversion fails."""
    if value is None or value == "None" or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_financial_metrics(ticker: str, end_date: Optional[str] = None, period: str = "ttm", limit: int = 10) -> List[FinancialMetrics]:
    """
    Get financial metrics from Alpha Vantage API.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        end_date (str, optional): End date in format 'YYYY-MM-DD'
        period (str, optional): Period type, e.g., 'ttm' for trailing twelve months or 'quarterly' for quarterly data
        limit (int, optional): Maximum number of data points to return
        
    Returns:
        List[FinancialMetrics]: List of financial metrics objects
    """
    # Replace with your Alpha Vantage API key
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    
    # Alpha Vantage endpoints
    overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    income_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
    balance_url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
    cashflow_url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={api_key}"
    
    print("balance_url", balance_url)
    # Fetch data from Alpha Vantage API
    try:
        overview_data = requests.get(overview_url).json()
        time.sleep(0.2)  # Avoid hitting rate limits
        income_data = requests.get(income_url).json()
        time.sleep(0.2)
        balance_data = requests.get(balance_url).json()
        time.sleep(0.2)
        cashflow_data = requests.get(cashflow_url).json()
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage: {e}")
        return []
    
    # Check for API errors
    if "Error Message" in overview_data:
        print(f"Alpha Vantage API error: {overview_data['Error Message']}")
        return []
    
    # Process data and create metrics objects
    results = []
    
    # Determine if we should use quarterly or annual reports
    # Note: 'quarterly' is more explicit than 'ttm' for quarterly data
    use_quarterly = period.lower() in ["quarterly", "ttm"]
    
    # Get reports based on period
    annual_reports = income_data.get("annualReports", [])
    quarterly_reports = income_data.get("quarterlyReports", [])
    reports_to_use = quarterly_reports if use_quarterly else annual_reports
    
    # Also get the corresponding balance and cashflow reports
    balance_reports = balance_data.get("quarterlyReports", []) if use_quarterly else balance_data.get("annualReports", [])
    cashflow_reports = cashflow_data.get("quarterlyReports", []) if use_quarterly else cashflow_data.get("annualReports", [])
    
    # Limit the number of reports
    reports_to_use = reports_to_use[:limit]
    
    # Get current share price from overview
    share_price = safe_float(overview_data.get("Price", 0))
    if share_price == 0:
        # Estimate from market cap and shares outstanding
        shares_outstanding = safe_float(overview_data.get("SharesOutstanding", 0))
        market_cap = safe_float(overview_data.get("MarketCapitalization", 0))
        if shares_outstanding > 0 and market_cap > 0:
            share_price = market_cap / shares_outstanding
    
    # Get base market data from overview
    base_market_cap = safe_float(overview_data.get("MarketCapitalization", 0))
    shares_outstanding = safe_float(overview_data.get("SharesOutstanding", 0))
    currency = overview_data.get("Currency", "USD")
    
    # Process each report
    for report in reports_to_use:
        report_date = report.get("fiscalDateEnding", "")
        
        # Skip if report date is after end_date
        if end_date and report_date > end_date:
            continue
            
        # Find matching balance sheet and cash flow data for this report period
        matching_balance = next((b for b in balance_reports 
                              if b.get("fiscalDateEnding") == report_date), {})
        matching_cashflow = next((c for c in cashflow_reports 
                               if c.get("fiscalDateEnding") == report_date), {})
        
        if not matching_balance or not matching_cashflow:
            print(f"Warning: Missing data for {report_date}, skipping...")
            continue
        
        try:
            # Basic financial values with safe conversion
            revenue = safe_float(report.get("totalRevenue"))
            ebitda = safe_float(report.get("ebitda"))
            net_income = safe_float(report.get("netIncome"))
            operating_income = safe_float(report.get("operatingIncome"))
            gross_profit = safe_float(report.get("grossProfit"))
            cost_of_revenue = safe_float(report.get("costOfRevenue"))
            interest_expense = safe_float(report.get("interestExpense"))
            
            total_assets = safe_float(matching_balance.get("totalAssets"))
            total_equity = safe_float(matching_balance.get("totalShareholderEquity"))
            total_current_assets = safe_float(matching_balance.get("totalCurrentAssets"))
            total_current_liabilities = safe_float(matching_balance.get("totalCurrentLiabilities"))
            inventory = safe_float(matching_balance.get("inventory"))
            receivables = safe_float(matching_balance.get("currentNetReceivables"))
            total_debt = safe_float(matching_balance.get("shortLongTermDebtTotal"))
            cash = safe_float(matching_balance.get("cashAndShortTermInvestments"))
            
            operating_cashflow = safe_float(matching_cashflow.get("operatingCashflow"))
            capital_expenditures = safe_float(matching_cashflow.get("capitalExpenditures"))
            free_cash_flow = operating_cashflow - capital_expenditures
            
            # EPS - try multiple sources
            eps = safe_float(report.get("reportedEPS"))
            if eps == 0 and net_income > 0 and shares_outstanding > 0:
                eps = net_income / shares_outstanding
            
            # Calculate period-specific market cap (adjust if historical)
            # This is an approximation - in a real system, you'd use historical price data
            current_year = datetime.now().year
            report_year = int(report_date.split('-')[0])
            years_diff = current_year - report_year
            
            # Adjust market cap if historical (simplistic approach)
            if years_diff > 0 and eps > 0:
                # Assume P/E ratio has been somewhat consistent
                current_pe = safe_float(overview_data.get("PERatio", 25))  # Default to 25 if not available
                market_cap = eps * shares_outstanding * current_pe
            else:
                market_cap = base_market_cap
            
            # Enterprise value
            enterprise_value = market_cap + total_debt - cash
            
            # Calculate financial ratios
            pe_ratio = market_cap / net_income if net_income > 0 else safe_float(overview_data.get("PERatio", 0))
            pb_ratio = market_cap / total_equity if total_equity > 0 else safe_float(overview_data.get("PriceToBookRatio", 0))
            ps_ratio = market_cap / revenue if revenue > 0 else safe_float(overview_data.get("PriceToSalesRatioTTM", 0))
            
            book_value_per_share = total_equity / shares_outstanding if shares_outstanding > 0 else 0
            free_cash_flow_per_share = free_cash_flow / shares_outstanding if shares_outstanding > 0 else 0
            
            # Calculate ROIC (Return on Invested Capital)
            # ROIC = NOPAT / Invested Capital
            # NOPAT = Operating Income * (1 - Tax Rate)
            # Invested Capital = Total Assets - Current Liabilities
            
            # Estimate tax rate (if available)
            income_tax = safe_float(report.get("incomeTaxExpense"))
            income_before_tax = safe_float(report.get("incomeBeforeTax"))
            tax_rate = income_tax / income_before_tax if income_before_tax > 0 else 0.25  # Default to 25% if not available
            
            # Calculate NOPAT
            nopat = operating_income * (1 - tax_rate)
            
            # Calculate Invested Capital
            invested_capital = total_assets - total_current_liabilities
            
            # Calculate ROIC
            roic = nopat / invested_capital if invested_capital > 0 else 0
            
            # Create financial metrics object
            metrics = FinancialMetrics(
                ticker=ticker,
                report_period=report_date,
                period=period,
                currency=currency,
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                price_to_earnings_ratio=pe_ratio,
                price_to_book_ratio=pb_ratio,
                price_to_sales_ratio=ps_ratio,
                enterprise_value_to_ebitda_ratio=enterprise_value / ebitda if ebitda > 0 else 0,
                enterprise_value_to_revenue_ratio=enterprise_value / revenue if revenue > 0 else 0,
                free_cash_flow_yield=free_cash_flow / market_cap if market_cap > 0 else 0,
                peg_ratio=safe_float(overview_data.get("PEGRatio", 0)),
                gross_margin=gross_profit / revenue if revenue > 0 else 0,
                operating_margin=operating_income / revenue if revenue > 0 else 0,
                net_margin=net_income / revenue if revenue > 0 else 0,
                return_on_equity=net_income / total_equity if total_equity > 0 else 0,
                return_on_assets=net_income / total_assets if total_assets > 0 else 0,
                return_on_invested_capital=roic,  # Use the calculated ROIC
                asset_turnover=revenue / total_assets if total_assets > 0 else 0,
                inventory_turnover=cost_of_revenue / inventory if inventory > 0 else 0,
                receivables_turnover=revenue / receivables if receivables > 0 else 0,
                days_sales_outstanding=365 * (receivables / revenue) if revenue > 0 and receivables > 0 else 0,
                operating_cycle=(365 * (inventory / cost_of_revenue) if cost_of_revenue > 0 and inventory > 0 else 0) + 
                               (365 * (receivables / revenue) if revenue > 0 and receivables > 0 else 0),
                working_capital_turnover=revenue / (total_current_assets - total_current_liabilities) 
                                      if (total_current_assets - total_current_liabilities) > 0 else 0,
                current_ratio=total_current_assets / total_current_liabilities if total_current_liabilities > 0 else 0,
                quick_ratio=(total_current_assets - inventory) / total_current_liabilities if total_current_liabilities > 0 else 0,
                cash_ratio=cash / total_current_liabilities if total_current_liabilities > 0 else 0,
                operating_cash_flow_ratio=operating_cashflow / total_current_liabilities if total_current_liabilities > 0 else 0,
                debt_to_equity=total_debt / total_equity if total_equity > 0 else 0,
                debt_to_assets=total_debt / total_assets if total_assets > 0 else 0,
                interest_coverage=operating_income / interest_expense if interest_expense > 0 else None,
                revenue_growth=0,  # Calculated later
                earnings_growth=0,  # Calculated later
                book_value_growth=0,  # Calculated later
                earnings_per_share_growth=0,  # Calculated later
                free_cash_flow_growth=0,  # Calculated later
                operating_income_growth=0,  # Calculated later
                ebitda_growth=0,  # Calculated later
                payout_ratio=safe_float(overview_data.get("PayoutRatio", 0)),
                earnings_per_share=eps,
                book_value_per_share=book_value_per_share,
                free_cash_flow_per_share=free_cash_flow_per_share,
            )
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error calculating metrics for {report_date}: {e}")
            continue
    
    # Sort results by report_period in descending order (most recent first)
    results.sort(key=lambda x: x.report_period, reverse=True)
    
    # Calculate growth metrics by comparing sequential periods
    for i in range(1, len(results)):
        try:
            current = results[i]  # Older period
            previous = results[i-1]  # More recent period
            
            # Note: Growth is calculated as (old - new) / new for YoY or sequential comparisons
            # This gives negative values for actual growth (which makes sense in financial terms)
            
            # Calculate period-over-period growth rates for raw values
            current.revenue_growth = (current.revenue_growth - previous.revenue_growth) / previous.revenue_growth if previous.revenue_growth else 0
            current.earnings_growth = (current.earnings_growth - previous.earnings_growth) / previous.earnings_growth if previous.earnings_growth else 0
            
            # Per-share metrics growth
            current.book_value_growth = (current.book_value_per_share - previous.book_value_per_share) / previous.book_value_per_share if previous.book_value_per_share else 0
            current.earnings_per_share_growth = (current.earnings_per_share - previous.earnings_per_share) / previous.earnings_per_share if previous.earnings_per_share else 0
            current.free_cash_flow_growth = (current.free_cash_flow_per_share - previous.free_cash_flow_per_share) / previous.free_cash_flow_per_share if previous.free_cash_flow_per_share else 0
            
            # Other financial metrics growth
            current.operating_income_growth = (current.operating_margin - previous.operating_margin) / previous.operating_margin if previous.operating_margin else 0
            current.ebitda_growth = 0  # Need EBITDA values from sequential periods
            
        except Exception as e:
            print(f"Error calculating growth metrics: {e}")
            continue
    
    return results


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch financial line items from Alpha Vantage API and calculate financial metrics.
    
    Args:
        ticker: Stock ticker symbol
        line_items: List of financial metrics to retrieve
        end_date: End date for data retrieval (YYYY-MM-DD)
        period: 'ttm' for trailing twelve months or 'annual' for annual reports
        limit: Maximum number of reports to return
        
    Returns:
        List of LineItem objects containing the requested financial metrics
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    
    # Fetch income statement data
    url_income = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}"
    response_income = requests.get(url_income)
    data_income = response_income.json()
    
    # Fetch balance sheet data
    url_balance = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}"
    response_balance = requests.get(url_balance)
    data_balance = response_balance.json()

    # Fetch cash flow statement data
    url_cashflow = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={api_key}"
    response_cashflow = requests.get(url_cashflow)
    data_cashflow = response_cashflow.json()

    # Fetch overview data   
    url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    response_overview = requests.get(url_overview)
    data_overview = response_overview.json()
    
    # Select annual or quarterly reports based on period parameter
    if period == 'annual':
        income_reports = data_income.get('annualReports', [])
        balance_reports = data_balance.get('annualReports', [])
        cashflow_reports = data_cashflow.get('annualReports', [])
    else:
        income_reports = data_income.get('quarterlyReports', [])
        balance_reports = data_balance.get('quarterlyReports', [])
        cashflow_reports = data_cashflow.get('quarterlyReports', [])
    # Convert to DataFrames
    df_income = pd.DataFrame(income_reports)
    df_balance = pd.DataFrame(balance_reports)
    df_cashflow = pd.DataFrame(cashflow_reports)
    # Filter by end date
    df_income = df_income[df_income['fiscalDateEnding'] <= end_date]
    df_balance = df_balance[df_balance['fiscalDateEnding'] <= end_date]
    df_cashflow = df_cashflow[df_cashflow['fiscalDateEnding'] <= end_date]
    # Merge datasets
    merged_data = pd.merge(
        df_income, 
        df_balance[[col for col in df_balance.columns if col not in df_income.columns or col == 'fiscalDateEnding']],
        on='fiscalDateEnding', 
        how='inner'
    )
    merged_data = pd.merge(
        merged_data,
        df_cashflow[[col for col in df_cashflow.columns if col not in merged_data.columns or col == 'fiscalDateEnding']],
        on='fiscalDateEnding',
        how='inner'
    )
    
    # Convert numeric columns to float if possible
    numeric_data = merged_data
    
    for col in merged_data.columns:
        try:
            merged_data[col] = pd.to_numeric(merged_data[col])
        except:
            pass
    
    # Calculate financial metrics
    numeric_data['taux_d_imposition_effectif'] = numeric_data['incomeTaxExpense'] / numeric_data['incomeBeforeTax']
    numeric_data['NOPAT'] = numeric_data['ebit'] * (1 - numeric_data['taux_d_imposition_effectif'])

    #numeric_data['goodwill_and_intangible_assets'] = numeric_data['goodwill'] + numeric_data['intangibleAssets']
    
    # Handle invalid tax rates
    numeric_data['NOPAT'] = numeric_data['NOPAT'].where(
        numeric_data['taux_d_imposition_effectif'] >= 0, 
        numeric_data['ebit']
    )
    
    # Calculate Invested Capital
    numeric_data['Invested_Capital'] = (
        numeric_data['totalAssets'] - 
        numeric_data['totalCurrentLiabilities']
    )
    
    # Calculate ROIC
    numeric_data['ROIC'] = numeric_data['NOPAT'] / numeric_data['Invested_Capital']
    
    # Limit number of results
    numeric_data = numeric_data.head(limit)
    numeric_data.replace(to_replace=['None', None], value=0, inplace=True)

    # Convert to LineItem objects
    results = []
    for _, row in numeric_data.iterrows():
        try:
            item = LineItem(
                ticker=ticker,
                report_period=row['fiscalDateEnding'],
                period=period,
                currency=row.get('reportedCurrency_x', 'USD'),
                return_on_invested_capital=row.get('ROIC'),
                operating_margin=float(row.get('operatingIncome', 0)) / float(row.get('totalRevenue', 1)) if float(row.get('totalRevenue', 0)) > 0 else 0,
                working_capital=float(row.get('totalCurrentAssets', 0)) - float(row.get('totalCurrentLiabilities', 0)),
                gross_margin=float(row.get('grossProfit', 0)) / float(row.get('totalRevenue', 1)) if float(row.get('totalRevenue', 0)) > 0 else 0,
                debt_to_equity=float(row.get('totalLiabilities', 0)) / float(row.get('totalShareholderEquity', 1)) if float(row.get('totalShareholderEquity', 0)) > 0 else 0,
                ebitda=float(row.get('ebitda', 0)),
                free_cash_flow=float(row.get('operatingCashflow', 0)) - float(row.get('capitalExpenditures', 0)),
                net_income=float(row.get('netIncome', 0)),
                capital_expenditure=float(row.get('capitalExpenditures', 0)),
                depreciation_and_amortization=float(row.get('depreciationAndAmortization', 0)),
                dividends_and_other_cash_distributions=float(row.get('dividendPayout', 0)),
                goodwill_and_intangible_assets=float(row.get('goodwill', 0)) + float(row.get('intangibleAssets', 0)),
                shareholders_equity=float(row.get('totalShareholderEquity', 0)),
                total_debt=float(row.get('shortLongTermDebtTotal', 0)),
                outstanding_shares=float(row.get('commonStockSharesOutstanding', 0)),
                total_assets=float(row.get('totalAssets', 0)),
                cash_and_equivalents=float(row.get('cashAndCashEquivalentsAtCarryingValue', 0)),
                total_liabilities=float(row.get('totalLiabilities', 0)),
                earnings_per_share=float(row.get('netIncome', 0)) / float(row.get('commonStockSharesOutstanding', 1)) if float(row.get('commonStockSharesOutstanding', 0)) > 0 else 0,
                revenue=float(row.get('totalRevenue', 0)),
                operating_income=float(row.get('operatingIncome', 0)),
                operating_expense=float(row.get('operatingExpenses', 0)),
                ebit=float(row.get('ebit', 0)),
                research_and_development=float(row.get('researchAndDevelopment', 0)),
                book_value_per_share=float(data_overview.get('BookValue', 0)),
                current_assets=float(row.get('totalCurrentAssets', 0)),
                current_liabilities=float(row.get('totalCurrentLiabilities', 0)),
            )
            results.append(item)
        except Exception as e:
            print(f"Error creating LineItem for {row['fiscalDateEnding']}: {e}")
            continue
    return results

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news



def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
