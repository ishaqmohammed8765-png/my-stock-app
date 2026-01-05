import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import numpy as np

class StockCalculatorPro:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Calculator Pro")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Storage for current stock data
        self.current_ticker = None
        self.current_price = None
        self.volatility = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the complete user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="📈 Stock Calculator Pro",
            font=("Arial", 24, "bold"),
            fg="#2c3e50"
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # ============ INPUT SECTION ============
        input_frame = ttk.LabelFrame(main_frame, text="Stock Lookup", padding="15")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Stock Ticker
        ttk.Label(input_frame, text="Stock Ticker:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.stock_ticker_entry = ttk.Entry(input_frame, font=("Arial", 11))
        self.stock_ticker_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        self.stock_ticker_entry.bind('<Return>', lambda e: self.lookup_stock_price())
        
        # Lookup Button
        lookup_btn = ttk.Button(
            input_frame,
            text="🔍 Lookup Price",
            command=self.lookup_stock_price
        )
        lookup_btn.grid(row=0, column=2, padx=(10, 0), pady=5)
        
        # ============ RESULTS SECTION ============
        results_frame = ttk.LabelFrame(main_frame, text="Stock Information", padding="15")
        results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(1, weight=1)
        
        # Current Price
        ttk.Label(results_frame, text="Current Price:", font=("Arial", 10)).grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.current_price_label = tk.Label(
            results_frame,
            text="--",
            font=("Arial", 14, "bold"),
            fg="#27ae60"
        )
        self.current_price_label.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Volatility
        ttk.Label(results_frame, text="Annualized Volatility:", font=("Arial", 10)).grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.volatility_label = tk.Label(
            results_frame,
            text="--",
            font=("Arial", 12)
        )
        self.volatility_label.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Buy Probability
        ttk.Label(results_frame, text="Probability (Reach Buy):", font=("Arial", 10)).grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.buy_prob_label = tk.Label(
            results_frame,
            text="--",
            font=("Arial", 12),
            fg="#3498db"
        )
        self.buy_prob_label.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Sell Probability
        ttk.Label(results_frame, text="Probability (Reach Sell):", font=("Arial", 10)).grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.sell_prob_label = tk.Label(
            results_frame,
            text="--",
            font=("Arial", 12),
            fg="#e74c3c"
        )
        self.sell_prob_label.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # ============ TRADING SECTION ============
        trading_frame = ttk.LabelFrame(main_frame, text="Trading Calculator", padding="15")
        trading_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        trading_frame.columnconfigure(1, weight=1)
        trading_frame.columnconfigure(3, weight=1)
        
        # Buy Price
        ttk.Label(trading_frame, text="Buy Price ($):", font=("Arial", 10)).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.buy_price_entry = ttk.Entry(trading_frame, font=("Arial", 11))
        self.buy_price_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 20))
        
        # Sell Price
        ttk.Label(trading_frame, text="Sell Price ($):", font=("Arial", 10)).grid(
            row=0, column=2, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.sell_price_entry = ttk.Entry(trading_frame, font=("Arial", 11))
        self.sell_price_entry.grid(row=0, column=3, sticky=(tk.W, tk.E), pady=5)
        
        # Quantity
        ttk.Label(trading_frame, text="Quantity:", font=("Arial", 10)).grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.quantity_entry = ttk.Entry(trading_frame, font=("Arial", 11))
        self.quantity_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(0, 20))
        self.quantity_entry.insert(0, "100")
        
        # Time Horizon (days)
        ttk.Label(trading_frame, text="Time Horizon (days):", font=("Arial", 10)).grid(
            row=1, column=2, sticky=tk.W, padx=(0, 10), pady=5
        )
        self.time_horizon_entry = ttk.Entry(trading_frame, font=("Arial", 11))
        self.time_horizon_entry.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=5)
        self.time_horizon_entry.insert(0, "30")
        
        # Buttons
        button_frame = ttk.Frame(trading_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=(10, 0))
        
        calc_btn = ttk.Button(
            button_frame,
            text="💰 Calculate P&L",
            command=self.calculate_profit_loss
        )
        calc_btn.pack(side=tk.LEFT, padx=5)
        
        add_btn = ttk.Button(
            button_frame,
            text="➕ Add to Portfolio",
            command=self.add_to_portfolio
        )
        add_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear All",
            command=self.clear_all
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Profit/Loss Display
        self.profit_label = tk.Label(
            trading_frame,
            text="",
            font=("Arial", 14, "bold")
        )
        self.profit_label.grid(row=3, column=0, columnspan=4, pady=(10, 0))
        
        # ============ PORTFOLIO SECTION ============
        portfolio_frame = ttk.LabelFrame(main_frame, text="Portfolio History", padding="10")
        portfolio_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        portfolio_frame.columnconfigure(0, weight=1)
        portfolio_frame.rowconfigure(0, weight=1)
        
        # Configure main grid to allow portfolio to expand
        main_frame.rowconfigure(4, weight=1)
        
        # Treeview with scrollbar
        tree_scroll = ttk.Scrollbar(portfolio_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.portfolio_table = ttk.Treeview(
            portfolio_frame,
            columns=("Ticker", "Date", "Buy", "Sell", "Qty", "P&L"),
            show="headings",
            height=8,
            yscrollcommand=tree_scroll.set
        )
        tree_scroll.config(command=self.portfolio_table.yview)
        
        # Define headings
        self.portfolio_table.heading("Ticker", text="Ticker")
        self.portfolio_table.heading("Date", text="Date/Time")
        self.portfolio_table.heading("Buy", text="Buy Price")
        self.portfolio_table.heading("Sell", text="Sell Price")
        self.portfolio_table.heading("Qty", text="Quantity")
        self.portfolio_table.heading("P&L", text="Profit/Loss")
        
        # Define column widths
        self.portfolio_table.column("Ticker", width=80, anchor=tk.CENTER)
        self.portfolio_table.column("Date", width=150, anchor=tk.CENTER)
        self.portfolio_table.column("Buy", width=100, anchor=tk.CENTER)
        self.portfolio_table.column("Sell", width=100, anchor=tk.CENTER)
        self.portfolio_table.column("Qty", width=80, anchor=tk.CENTER)
        self.portfolio_table.column("P&L", width=120, anchor=tk.CENTER)
        
        self.portfolio_table.pack(fill=tk.BOTH, expand=True)
        
        # Delete button for portfolio
        delete_btn = ttk.Button(
            main_frame,
            text="❌ Delete Selected",
            command=self.delete_selected
        )
        delete_btn.grid(row=5, column=0, columnspan=3, pady=(5, 0))
    
    def calculate_historical_volatility(self, ticker_symbol, days=60):
        """
        Calculate annualized historical volatility using actual price data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer
            
            # Fetch historical data
            ticker = yf.Ticker(ticker_symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty or len(hist_data) < 20:
                return None, "Insufficient historical data"
            
            # Calculate log returns
            hist_data['Log_Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            
            # Remove NaN values
            log_returns = hist_data['Log_Return'].dropna()
            
            if len(log_returns) < 20:
                return None, "Insufficient data for volatility calculation"
            
            # Calculate annualized volatility
            # Assuming 252 trading days per year
            daily_volatility = log_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            return annualized_volatility, None
            
        except Exception as e:
            return None, f"Error calculating volatility: {str(e)}"
    
    def calculate_lognormal_probability(self, S0, ST, sigma, T, mu=0.0):
        """
        Calculate probability of reaching target price using Log-Normal distribution
        
        Parameters:
        S0: Current stock price
        ST: Target stock price
        sigma: Annualized volatility
        T: Time horizon in years
        mu: Expected return (default 0 for risk-neutral)
        
        Returns:
        Probability of stock price being above ST at time T
        """
        if T <= 0 or sigma <= 0 or S0 <= 0:
            return 0.5  # Return neutral probability if invalid inputs
        
        try:
            # Log-normal model: ln(ST/S0) ~ N((mu - sigma^2/2)*T, sigma*sqrt(T))
            d = (np.log(ST / S0) - (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            # P(S > ST) = 1 - N(d)
            prob_above = 1 - norm.cdf(d)
            
            return prob_above
            
        except Exception as e:
            return 0.5
    
    def lookup_stock_price(self):
        """Fetch stock price and calculate volatility with robust error handling"""
        try:
            # Get ticker symbol
            ticker_symbol = self.stock_ticker_entry.get().strip().upper()
            
            if not ticker_symbol:
                messagebox.showwarning("Input Error", "Please enter a stock ticker symbol")
                return
            
            # Show loading state
            self.current_price_label.config(text="Loading...", fg="#95a5a6")
            self.root.update()
            
            # Fetch stock data
            ticker = yf.Ticker(ticker_symbol)
            
            # Try to get current price using fast_info first
            try:
                current_price = ticker.fast_info['lastPrice']
            except:
                # Fallback to history
                hist = ticker.history(period="1d")
                if hist.empty:
                    raise ValueError("No price data available")
                current_price = hist['Close'].iloc[-1]
            
            if pd.isna(current_price) or current_price <= 0:
                raise ValueError("Invalid price data")
            
            # Calculate historical volatility
            volatility, error = self.calculate_historical_volatility(ticker_symbol)
            
            if error:
                messagebox.showwarning("Volatility Warning", error)
                volatility = 0.3  # Default to 30% if calculation fails
            
            # Store current data
            self.current_ticker = ticker_symbol
            self.current_price = current_price
            self.volatility = volatility
            
            # Update display
            self.current_price_label.config(
                text=f"${current_price:.2f}",
                fg="#27ae60"
            )
            self.volatility_label.config(
                text=f"{volatility*100:.2f}%"
            )
            
            # Calculate suggested buy/sell prices (5% below/above)
            suggested_buy = current_price * 0.95
            suggested_sell = current_price * 1.05
            
            # Populate entry fields
            self.buy_price_entry.delete(0, tk.END)
            self.buy_price_entry.insert(0, f"{suggested_buy:.2f}")
            
            self.sell_price_entry.delete(0, tk.END)
            self.sell_price_entry.insert(0, f"{suggested_sell:.2f}")
            
            # Calculate probabilities
            self.update_probabilities()
            
            messagebox.showinfo(
                "Success",
                f"Stock data loaded for {ticker_symbol}\n"
                f"Price: ${current_price:.2f}\n"
                f"Volatility: {volatility*100:.2f}%"
            )
            
        except ValueError as ve:
            self.current_price_label.config(text="Error", fg="#e74c3c")
            messagebox.showerror("Data Error", str(ve))
        except Exception as e:
            self.current_price_label.config(text="Error", fg="#e74c3c")
            error_msg = str(e)
            if "No data found" in error_msg or "404" in error_msg:
                messagebox.showerror(
                    "Ticker Not Found",
                    f"Could not find ticker '{self.stock_ticker_entry.get()}'.\n"
                    "Please check the symbol and try again."
                )
            elif "timed out" in error_msg.lower() or "connection" in error_msg.lower():
                messagebox.showerror(
                    "Network Error",
                    "Connection error. Please check your internet connection."
                )
            else:
                messagebox.showerror("Error", f"An error occurred: {error_msg}")
    
    def update_probabilities(self):
        """Update probability calculations based on current inputs"""
        try:
            if self.current_price is None or self.volatility is None:
                return
            
            buy_price = float(self.buy_price_entry.get())
            sell_price = float(self.sell_price_entry.get())
            time_horizon = float(self.time_horizon_entry.get())
            
            # Convert days to years
            T = time_horizon / 365.0
            
            # Calculate probabilities using log-normal model
            # For buy price (below current): probability of going down
            if buy_price < self.current_price:
                buy_prob = 1 - self.calculate_lognormal_probability(
                    self.current_price, buy_price, self.volatility, T
                )
            else:
                buy_prob = self.calculate_lognormal_probability(
                    self.current_price, buy_price, self.volatility, T
                )
            
            # For sell price (above current): probability of going up
            if sell_price > self.current_price:
                sell_prob = self.calculate_lognormal_probability(
                    self.current_price, sell_price, self.volatility, T
                )
            else:
                sell_prob = 1 - self.calculate_lognormal_probability(
                    self.current_price, sell_price, self.volatility, T
                )
            
            # Update labels
            self.buy_prob_label.config(text=f"{buy_prob:.2%}")
            self.sell_prob_label.config(text=f"{sell_prob:.2%}")
            
        except ValueError:
            # If inputs are invalid, clear probability display
            self.buy_prob_label.config(text="--")
            self.sell_prob_label.config(text="--")
    
    def calculate_profit_loss(self):
        """Calculate profit/loss based on buy/sell prices and quantity"""
        try:
            buy_price = float(self.buy_price_entry.get())
            sell_price = float(self.sell_price_entry.get())
            quantity = int(self.quantity_entry.get())
            
            if buy_price <= 0 or sell_price <= 0 or quantity <= 0:
                raise ValueError("Prices and quantity must be positive")
            
            # Calculate totals
            total_buy_cost = buy_price * quantity
            total_sell_revenue = sell_price * quantity
            profit_loss = total_sell_revenue - total_buy_cost
            profit_loss_pct = (profit_loss / total_buy_cost) * 100
            
            # Update display
            if profit_loss > 0:
                self.profit_label.config(
                    text=f"💰 Profit: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)",
                    fg="#27ae60"
                )
            elif profit_loss < 0:
                self.profit_label.config(
                    text=f"📉 Loss: ${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%)",
                    fg="#e74c3c"
                )
            else:
                self.profit_label.config(
                    text=f"Break Even: $0.00",
                    fg="#95a5a6"
                )
            
            # Update probabilities
            self.update_probabilities()
            
        except ValueError as e:
            self.profit_label.config(
                text=f"⚠️ Invalid input: {str(e)}",
                fg="#e74c3c"
            )
            messagebox.showerror("Input Error", str(e))
    
    def add_to_portfolio(self):
        """Add current trade to portfolio table"""
        try:
            # Validate all inputs
            if not self.current_ticker:
                raise ValueError("Please lookup a stock first")
            
            buy_price = float(self.buy_price_entry.get())
            sell_price = float(self.sell_price_entry.get())
            quantity = int(self.quantity_entry.get())
            
            if buy_price <= 0 or sell_price <= 0 or quantity <= 0:
                raise ValueError("Prices and quantity must be positive")
            
            # Calculate P&L
            profit_loss = (sell_price - buy_price) * quantity
            
            # Get current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format values
            pl_formatted = f"${profit_loss:+.2f}"
            
            # Insert into table
            self.portfolio_table.insert(
                "",
                "end",
                values=(
                    self.current_ticker,
                    current_time,
                    f"${buy_price:.2f}",
                    f"${sell_price:.2f}",
                    quantity,
                    pl_formatted
                ),
                tags=('profit' if profit_loss >= 0 else 'loss',)
            )
            
            # Configure tags for color coding
            self.portfolio_table.tag_configure('profit', foreground='#27ae60')
            self.portfolio_table.tag_configure('loss', foreground='#e74c3c')
            
            messagebox.showinfo("Success", "Trade added to portfolio!")
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def delete_selected(self):
        """Delete selected row from portfolio"""
        selected_items = self.portfolio_table.selection()
        if not selected_items:
            messagebox.showwarning("Selection Error", "Please select a row to delete")
            return
        
        for item in selected_items:
            self.portfolio_table.delete(item)
    
    def clear_all(self):
        """Clear all input fields and results"""
        response = messagebox.askyesno(
            "Clear All",
            "Are you sure you want to clear all inputs and results?"
        )
        
        if response:
            # Clear entries
            self.stock_ticker_entry.delete(0, tk.END)
            self.buy_price_entry.delete(0, tk.END)
            self.sell_price_entry.delete(0, tk.END)
            self.quantity_entry.delete(0, tk.END)
            self.quantity_entry.insert(0, "100")
            self.time_horizon_entry.delete(0, tk.END)
            self.time_horizon_entry.insert(0, "30")
            
            # Clear labels
            self.current_price_label.config(text="--")
            self.volatility_label.config(text="--")
            self.buy_prob_label.config(text="--")
            self.sell_prob_label.config(text="--")
            self.profit_label.config(text="")
            
            # Clear stored data
            self.current_ticker = None
            self.current_price = None
            self.volatility = None

def main():
    root = tk.Tk()
    app = StockCalculatorPro(root)
    root.mainloop()

if __name__ == "__main__":
    main()
