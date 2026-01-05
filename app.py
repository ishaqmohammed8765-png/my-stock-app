import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from scipy.stats import norm
import numpy as np
import threading

class StockCalculatorPro:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Calculator Pro")
        self.root.geometry("950x750")
        
        # Storage for current stock data
        self.current_ticker = None
        self.current_price = None
        self.volatility = None
        self.stock_data = {}
        
        # Configure style
        try:
            self.style = ttk.Style()
            self.style.theme_use('clam')
        except:
            pass  # Use default if clam not available
        
        # Initialize UI immediately
        self.setup_ui()
        
        # Focus on ticker entry
        self.stock_ticker_entry.focus_set()
    
    def setup_ui(self):
        """Create the complete user interface"""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#f5f6fa')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ============ TITLE ============
        title_frame = tk.Frame(main_frame, bg='#2c3e50', relief=tk.RAISED, borderwidth=2)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            title_frame,
            text="📈 Stock Calculator Pro",
            font=("Arial", 26, "bold"),
            fg="white",
            bg='#2c3e50',
            pady=15
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Advanced Trading Analysis with Log-Normal Probability Models",
            font=("Arial", 10),
            fg="#ecf0f1",
            bg='#2c3e50',
            pady=(0, 10)
        )
        subtitle_label.pack()
        
        # ============ INPUT SECTION ============
        input_frame = tk.LabelFrame(
            main_frame,
            text=" 🔍 Stock Lookup ",
            font=("Arial", 11, "bold"),
            bg='#f5f6fa',
            fg='#2c3e50',
            relief=tk.GROOVE,
            borderwidth=2,
            padx=20,
            pady=15
        )
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Ticker input row
        ticker_frame = tk.Frame(input_frame, bg='#f5f6fa')
        ticker_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            ticker_frame,
            text="Stock Ticker:",
            font=("Arial", 11, "bold"),
            bg='#f5f6fa',
            fg='#34495e'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.stock_ticker_entry = tk.Entry(
            ticker_frame,
            font=("Arial", 13),
            width=15,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.stock_ticker_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.stock_ticker_entry.bind('<Return>', lambda e: self.lookup_stock_price())
        
        lookup_btn = tk.Button(
            ticker_frame,
            text="🔍 Lookup Stock",
            command=self.lookup_stock_price,
            font=("Arial", 11, "bold"),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=5,
            cursor='hand2'
        )
        lookup_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_label = tk.Label(
            ticker_frame,
            text="",
            font=("Arial", 9),
            bg='#f5f6fa',
            fg='#7f8c8d'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # ============ STOCK INFO SECTION ============
        info_frame = tk.LabelFrame(
            main_frame,
            text=" 📊 Stock Information ",
            font=("Arial", 11, "bold"),
            bg='#f5f6fa',
            fg='#2c3e50',
            relief=tk.GROOVE,
            borderwidth=2,
            padx=20,
            pady=15
        )
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create grid for info display
        info_grid = tk.Frame(info_frame, bg='#f5f6fa')
        info_grid.pack(fill=tk.X)
        
        # Current Price
        price_frame = tk.Frame(info_grid, bg='#ecf0f1', relief=tk.RAISED, borderwidth=1)
        price_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        info_grid.columnconfigure(0, weight=1)
        
        tk.Label(
            price_frame,
            text="Current Price",
            font=("Arial", 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        ).pack(pady=(5, 0))
        
        self.current_price_label = tk.Label(
            price_frame,
            text="--",
            font=("Arial", 20, "bold"),
            bg='#ecf0f1',
            fg='#27ae60'
        )
        self.current_price_label.pack(pady=(0, 5))
        
        # Volatility
        vol_frame = tk.Frame(info_grid, bg='#ecf0f1', relief=tk.RAISED, borderwidth=1)
        vol_frame.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        info_grid.columnconfigure(1, weight=1)
        
        tk.Label(
            vol_frame,
            text="Ann. Volatility (σ)",
            font=("Arial", 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        ).pack(pady=(5, 0))
        
        self.volatility_label = tk.Label(
            vol_frame,
            text="--",
            font=("Arial", 16, "bold"),
            bg='#ecf0f1',
            fg='#e67e22'
        )
        self.volatility_label.pack(pady=(0, 5))
        
        # Buy Probability
        buy_prob_frame = tk.Frame(info_grid, bg='#ecf0f1', relief=tk.RAISED, borderwidth=1)
        buy_prob_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        tk.Label(
            buy_prob_frame,
            text="Buy Target Probability",
            font=("Arial", 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        ).pack(pady=(5, 0))
        
        self.buy_prob_label = tk.Label(
            buy_prob_frame,
            text="--",
            font=("Arial", 16, "bold"),
            bg='#ecf0f1',
            fg='#3498db'
        )
        self.buy_prob_label.pack(pady=(0, 5))
        
        # Sell Probability
        sell_prob_frame = tk.Frame(info_grid, bg='#ecf0f1', relief=tk.RAISED, borderwidth=1)
        sell_prob_frame.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        tk.Label(
            sell_prob_frame,
            text="Sell Target Probability",
            font=("Arial", 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        ).pack(pady=(5, 0))
        
        self.sell_prob_label = tk.Label(
            sell_prob_frame,
            text="--",
            font=("Arial", 16, "bold"),
            bg='#ecf0f1',
            fg='#e74c3c'
        )
        self.sell_prob_label.pack(pady=(0, 5))
        
        # ============ TRADING CALCULATOR ============
        trading_frame = tk.LabelFrame(
            main_frame,
            text=" 💰 Trading Calculator ",
            font=("Arial", 11, "bold"),
            bg='#f5f6fa',
            fg='#2c3e50',
            relief=tk.GROOVE,
            borderwidth=2,
            padx=20,
            pady=15
        )
        trading_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input grid
        inputs_grid = tk.Frame(trading_frame, bg='#f5f6fa')
        inputs_grid.pack(fill=tk.X, pady=5)
        
        # Row 1: Buy and Sell Price
        tk.Label(
            inputs_grid,
            text="Buy Price ($):",
            font=("Arial", 10, "bold"),
            bg='#f5f6fa',
            fg='#34495e'
        ).grid(row=0, column=0, sticky='w', pady=5)
        
        self.buy_price_entry = tk.Entry(
            inputs_grid,
            font=("Arial", 12),
            width=15,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.buy_price_entry.grid(row=0, column=1, sticky='ew', padx=10, pady=5)
        self.buy_price_entry.bind('<KeyRelease>', lambda e: self.update_probabilities())
        
        tk.Label(
            inputs_grid,
            text="Sell Price ($):",
            font=("Arial", 10, "bold"),
            bg='#f5f6fa',
            fg='#34495e'
        ).grid(row=0, column=2, sticky='w', padx=(20, 0), pady=5)
        
        self.sell_price_entry = tk.Entry(
            inputs_grid,
            font=("Arial", 12),
            width=15,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.sell_price_entry.grid(row=0, column=3, sticky='ew', padx=10, pady=5)
        self.sell_price_entry.bind('<KeyRelease>', lambda e: self.update_probabilities())
        
        # Row 2: Quantity and Time Horizon
        tk.Label(
            inputs_grid,
            text="Quantity:",
            font=("Arial", 10, "bold"),
            bg='#f5f6fa',
            fg='#34495e'
        ).grid(row=1, column=0, sticky='w', pady=5)
        
        self.quantity_entry = tk.Entry(
            inputs_grid,
            font=("Arial", 12),
            width=15,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.quantity_entry.grid(row=1, column=1, sticky='ew', padx=10, pady=5)
        self.quantity_entry.insert(0, "100")
        
        tk.Label(
            inputs_grid,
            text="Time Horizon (days):",
            font=("Arial", 10, "bold"),
            bg='#f5f6fa',
            fg='#34495e'
        ).grid(row=1, column=2, sticky='w', padx=(20, 0), pady=5)
        
        self.time_horizon_entry = tk.Entry(
            inputs_grid,
            font=("Arial", 12),
            width=15,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.time_horizon_entry.grid(row=1, column=3, sticky='ew', padx=10, pady=5)
        self.time_horizon_entry.insert(0, "30")
        self.time_horizon_entry.bind('<KeyRelease>', lambda e: self.update_probabilities())
        
        # Configure column weights
        inputs_grid.columnconfigure(1, weight=1)
        inputs_grid.columnconfigure(3, weight=1)
        
        # Buttons row
        button_frame = tk.Frame(trading_frame, bg='#f5f6fa')
        button_frame.pack(fill=tk.X, pady=(10, 5))
        
        calc_btn = tk.Button(
            button_frame,
            text="💰 Calculate P&L",
            command=self.calculate_profit_loss,
            font=("Arial", 10, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        calc_btn.pack(side=tk.LEFT, padx=5)
        
        add_btn = tk.Button(
            button_frame,
            text="➕ Add to Portfolio",
            command=self.add_to_portfolio,
            font=("Arial", 10, "bold"),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        add_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear All",
            command=self.clear_all,
            font=("Arial", 10, "bold"),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = tk.Button(
            button_frame,
            text="📥 Export Portfolio",
            command=self.export_portfolio,
            font=("Arial", 10, "bold"),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # P&L Display
        self.profit_label = tk.Label(
            trading_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg='#f5f6fa'
        )
        self.profit_label.pack(pady=(10, 0))
        
        # ============ PORTFOLIO TABLE ============
        portfolio_frame = tk.LabelFrame(
            main_frame,
            text=" 📈 Portfolio History ",
            font=("Arial", 11, "bold"),
            bg='#f5f6fa',
            fg='#2c3e50',
            relief=tk.GROOVE,
            borderwidth=2,
            padx=10,
            pady=10
        )
        portfolio_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview with scrollbars
        tree_frame = tk.Frame(portfolio_frame, bg='#f5f6fa')
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.portfolio_table = ttk.Treeview(
            tree_frame,
            columns=("Ticker", "Date", "Buy", "Sell", "Qty", "P&L", "P&L%"),
            show="headings",
            height=10,
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        
        tree_scroll_y.config(command=self.portfolio_table.yview)
        tree_scroll_x.config(command=self.portfolio_table.xview)
        
        # Define headings
        self.portfolio_table.heading("Ticker", text="Ticker")
        self.portfolio_table.heading("Date", text="Date/Time")
        self.portfolio_table.heading("Buy", text="Buy Price")
        self.portfolio_table.heading("Sell", text="Sell Price")
        self.portfolio_table.heading("Qty", text="Quantity")
        self.portfolio_table.heading("P&L", text="Profit/Loss ($)")
        self.portfolio_table.heading("P&L%", text="Return (%)")
        
        # Define column widths
        self.portfolio_table.column("Ticker", width=80, anchor=tk.CENTER)
        self.portfolio_table.column("Date", width=150, anchor=tk.CENTER)
        self.portfolio_table.column("Buy", width=100, anchor=tk.CENTER)
        self.portfolio_table.column("Sell", width=100, anchor=tk.CENTER)
        self.portfolio_table.column("Qty", width=80, anchor=tk.CENTER)
        self.portfolio_table.column("P&L", width=120, anchor=tk.CENTER)
        self.portfolio_table.column("P&L%", width=100, anchor=tk.CENTER)
        
        self.portfolio_table.pack(fill=tk.BOTH, expand=True)
        
        # Portfolio control buttons
        portfolio_btn_frame = tk.Frame(main_frame, bg='#f5f6fa')
        portfolio_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        delete_btn = tk.Button(
            portfolio_btn_frame,
            text="❌ Delete Selected",
            command=self.delete_selected,
            font=("Arial", 10, "bold"),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=5,
            cursor='hand2'
        )
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Summary label
        self.summary_label = tk.Label(
            portfolio_btn_frame,
            text="Total Trades: 0 | Net P&L: $0.00",
            font=("Arial", 10, "bold"),
            bg='#f5f6fa',
            fg='#34495e'
        )
        self.summary_label.pack(side=tk.RIGHT, padx=10)
    
    def calculate_historical_volatility(self, ticker_symbol, days=60):
        """Calculate annualized historical volatility using actual price data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)
            
            ticker = yf.Ticker(ticker_symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty or len(hist_data) < 20:
                return None, "Insufficient historical data (need 20+ trading days)"
            
            # Calculate log returns
            hist_data['Log_Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            log_returns = hist_data['Log_Return'].dropna()
            
            if len(log_returns) < 20:
                return None, "Insufficient data for volatility calculation"
            
            # Calculate annualized volatility (252 trading days)
            daily_volatility = log_returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            return annualized_volatility, None
            
        except Exception as e:
            return None, f"Volatility calculation error: {str(e)}"
    
    def calculate_lognormal_probability(self, S0, ST, sigma, T, mu=0.0):
        """
        Calculate probability using Log-Normal distribution
        
        P(S_T > ST) where ln(S_T/S_0) ~ N((mu - sigma^2/2)*T, sigma*sqrt(T))
        """
        if T <= 0 or sigma <= 0 or S0 <= 0 or ST <= 0:
            return 0.5
        
        try:
            # Log-normal model
            d = (np.log(ST / S0) - (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            prob_above = 1 - norm.cdf(d)
            return prob_above
        except:
            return 0.5
    
    def lookup_stock_price(self):
        """Fetch stock price with threading to prevent GUI freeze"""
        ticker_symbol = self.stock_ticker_entry.get().strip().upper()
        
        if not ticker_symbol:
            messagebox.showwarning("Input Error", "Please enter a stock ticker symbol")
            return
        
        # Show loading state
        self.status_label.config(text="⏳ Loading...", fg='#e67e22')
        self.current_price_label.config(text="Loading...", fg='#95a5a6')
        self.volatility_label.config(text="Loading...", fg='#95a5a6')
        self.root.update_idletasks()
        
        # Run in thread to prevent freezing
        thread = threading.Thread(target=self._fetch_stock_data, args=(ticker_symbol,))
        thread.daemon = True
        thread.start()
    
    def _fetch_stock_data(self, ticker_symbol):
        """Background thread for fetching stock data"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get current price
            try:
                current_price = ticker.fast_info['lastPrice']
            except:
                hist = ticker.history(period="1d")
                if hist.empty:
                    raise ValueError("No price data available")
                current_price = hist['Close'].iloc[-1]
            
            if pd.isna(current_price) or current_price <= 0:
                raise ValueError("Invalid price data")
            
            # Calculate volatility
            volatility, error = self.calculate_historical_volatility(ticker_symbol)
            
            if error:
                volatility = 0.3  # Default 30% if calculation fails
                vol_note = " (default)"
            else:
                vol_note = ""
            
            # Store data
            self.current_ticker = ticker_symbol
            self.current_price = current_price
            self.volatility = volatility
            
            # Update UI (must be done in main thread)
            self.root.after(0, self._update_stock_display, current_price, volatility, vol_note, None)
            
        except ValueError as ve:
            self.root.after(0, self._update_stock_display, None, None, None, str(ve))
        except Exception as e:
            error_msg = str(e)
            if "No data found" in error_msg or "404" in error_msg:
                error_msg = f"Ticker '{ticker_symbol}' not found"
            elif "timed out" in error_msg.lower():
                error_msg = "Connection timeout - check internet"
            self.root.after(0, self._update_stock_display, None, None, None, error_msg)
    
    def _update_stock_display(self, current_price, volatility, vol_note, error):
        """Update UI with fetched stock data (runs in main thread)"""
        if error:
            self.current_price_label.config(text="Error", fg='#e74c3c')
            self.volatility_label.config(text="--", fg='#95a5a6')
            self.status_label.config(text="❌ Failed", fg='#e74c3c')
            messagebox.showerror("Error", error)
            return
        
        # Update displays
        self.current_price_label.config(text=f"${current_price:.2f}", fg='#27ae60')
        self.volatility_label.config(text=f"{volatility*100:.2f}%{vol_note}", fg='#e67e22')
        self.status_label.config(text="✅ Success", fg='#27ae60')
        
        # Populate suggested prices (5% range)
        suggested_buy = current_price * 0.95
        suggested_sell = current_price * 1.05
        
        self.buy_price_entry.delete(0, tk.END)
        self.buy_price_entry.insert(0, f"{suggested_buy:.2f}")
        
        self.sell_price_entry.delete(0, tk.END)
        self.sell_price_entry.insert(0, f"{suggested_sell:.2f}")
        
        # Update probabilities
        self.update_probabilities()
        
        # Show success message
        messagebox.showinfo(
            "Stock Loaded",
            f"✅ {self.current_ticker}\n"
            f"Price: ${current_price:.2f}\n"
            f"Volatility: {volatility*100:.2f}%"
        )
    
    def update_probabilities(self):
        """Update probability calculations"""
        try:
            if self.current_price is None or self.volatility is None:
                return
            
            buy_price = float(self.buy_price_entry.get())
            sell_price = float(self.sell_price_entry.get())
            time_horizon = float(self.time_horizon_entry.get())
            
            T = time_horizon / 365.0
            
            # Calculate probabilities
            if buy_price < self.current_price:
                buy_prob = 1 - self.calculate_lognormal_probability(
                    self.current_price, buy_price, self.volatility, T
                )
            else:
                buy_prob = self.calculate_lognormal_probability(
                    self.current_price, buy_price, self.volatility, T
                )
            
            if sell_price > self.current_price:
                sell_prob = self.calculate_lognormal_probability(
                    self.current_price, sell_price, self.volatility, T
                )
            else:
                sell_prob = 1 - self.calculate_lognormal_probability(
                    self.current_price, sell_price, self.volatility, T
                )
            
            self.buy_prob_label.config(text=f"{buy_prob:.2%}")
            self.sell_prob_label.config(text=f"{sell_prob:.2%}")
            
        except ValueError:
            self.buy_prob_label.config(text="--")
            self.sell_prob_label.config(text="--")
    
    def calculate_profit_loss(self):
        """Calculate and display profit/loss"""
        try:
            buy_price = float(self.buy_price_entry.get())
            sell_price = float(self.sell_price_entry.get())
            quantity = int(self.quantity_entry.get())
            
            if buy_price <= 0 or sell_price <= 0 or quantity <= 0:
                raise ValueError("All values must be positive")
            
            total_cost = buy_price * quantity
            total_revenue = sell_price * quantity
            profit_loss = total_revenue - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100
            
            if profit_loss > 0:
                self.profit_label.config(
                    text=f"💰 Profit: ${profit_loss:,.2f} ({profit_loss_pct:+.2f}%)",
                    fg='#27ae60'
                )
            elif profit_loss < 0:
                self.profit_label.config(
                    text=f"📉 Loss: ${abs(profit_loss):,.2f} ({profit_loss_pct:.2f}%)",
                    fg='#e74c3c'
                )
            else:
                self.profit_label.config(text="⚖️ Break Even", fg='#95a5a6')
            
            self.update_probabilities()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
    
    def add_to_portfolio(self):
        """Add trade to portfolio table"""
        try:
            if not self.current_ticker:
                raise ValueError("Please lookup a stock first")
            
            buy_price = float(self.buy_price_entry.get())
            sell_price = float(self.sell_price_entry.get())
            quantity = int(self.quantity_entry.get())
            
            if buy_price <= 0 or sell_price <= 0 or quantity <= 0:
                raise ValueError("All values must be positive")
            
            profit_loss = (sell_price - buy_price) * quantity
            profit_loss_pct = ((sell_price - buy_price) / buy_price) * 100
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.portfolio_table.insert(
                "",
                "end",
                values=(
                    self.current_ticker,
                    current_time,
                    f"${buy_price:.2f}",
                    f"${sell_price:.2f}",
                    quantity,
                    f"${profit_loss:+,.2f}",
                    f"{profit_loss_pct:+.2f}%"
                ),
                tags=('profit' if profit_loss >= 0 else 'loss',)
            )
            
            self.portfolio_table.tag_configure('profit', foreground='#27ae60')
            self.portfolio_table.tag_configure('loss', foreground='#e74c3c')
            
            self.update_portfolio_summary()
            messagebox.showinfo("Success", "✅ Trade added to portfolio!")
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def update_portfolio_summary(self):
        """Update portfolio summary statistics"""
        total_trades = len(self.portfolio_table.get_children())
        total_pl = 0.0
        
        for item in self.portfolio_table.get_children():
            values = self.portfolio_table.item(item)['values']
            pl_str = values[5].replace('$', '').replace(',', '')
            total_pl += float(pl_str)
        
        self.summary_label.config(
            text=f"Total Trades: {total_trades} | Net P&L: ${total_pl:+,.2f}"
        )
    
    def delete_selected(self):
        """Delete selected rows from portfolio"""
        selected = self.portfolio_table.selection()
        if not selected:
            messagebox.showwarning("Selection Error", "Please select row(s) to delete")
            return
        
        for item in selected:
            self.portfolio_table.delete(item)
        
        self.update_portfolio_summary()
    
    def export_portfolio(self):
        """Export portfolio to CSV file"""
        if not self.portfolio_table.get_children():
            messagebox.showwarning("Export Error", "Portfolio is empty - nothing to export")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if filename:
                data = []
                for item in self.portfolio_table.get_children():
                    data.append(self.portfolio_table.item(item)['values'])
                
                df = pd.DataFrame(
                    data,
                    columns=["Ticker", "Date/Time", "Buy Price", "Sell Price", "Quantity", "P&L ($)", "Return (%)"]
                )
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Success", f"✅ Portfolio exported to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def clear_all(self):
        """Clear all inputs and reset"""
        response = messagebox.askyesno(
            "Clear All",
            "Clear all inputs and results?\n(Portfolio will remain)"
        )
        
        if response:
            self.stock_ticker_entry.delete(0, tk.END)
            self.buy_price_entry.delete(0, tk.END)
            self.sell_price_entry.delete(0, tk.END)
            self.quantity_entry.delete(0, tk.END)
            self.quantity_entry.insert(0, "100")
            self.time_horizon_entry.delete(0, tk.END)
            self.time_horizon_entry.insert(0, "30")
            
            self.current_price_label.config(text="--", fg='#95a5a6')
            self.volatility_label.config(text="--", fg='#95a5a6')
            self.buy_prob_label.config(text="--", fg='#95a5a6')
            self.sell_prob_label.config(text="--", fg='#95a5a6')
            self.profit_label.config(text="")
            self.status_label.config(text="")
            
            self.current_ticker = None
            self.current_price = None
            self.volatility = None

def main():
    """Main entry point - ensures GUI starts properly"""
    try:
        root = tk.Tk()
        app = StockCalculatorPro(root)
        root.mainloop()
    except Exception as e:
        print(f"Critical error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
