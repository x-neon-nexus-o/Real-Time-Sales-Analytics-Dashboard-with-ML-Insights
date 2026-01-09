"""
Real-Time Data Ingestion Module.
Simulates continuous sales data streaming with realistic patterns.
"""

import threading
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from collections import deque
import numpy as np
import pandas as pd

from config import (
    STREAMING_CONFIG, CATEGORIES, PRODUCTS, REGIONS, PAYMENT_METHODS,
    HOURLY_PATTERNS, DAILY_PATTERNS, MONTHLY_PATTERNS, RAW_DATA_DIR
)
from utils import (
    setup_logger, generate_transaction_id, generate_customer_id,
    round_currency
)

logger = setup_logger("data_ingestion")


class RealTimeDataGenerator:
    """
    Simulates real-time sales data streaming.
    
    Generates new transactions at configurable intervals with realistic
    time-based patterns. Maintains a thread-safe buffer of recent transactions.
    """
    
    def __init__(
        self,
        base_data: Optional[pd.DataFrame] = None,
        generation_rate: int = 5,
        buffer_size: int = 10000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the real-time data generator.
        
        Args:
            base_data: Optional historical data to learn patterns from
            generation_rate: Seconds between transaction generations
            buffer_size: Maximum transactions to keep in buffer
            random_seed: Optional random seed for reproducibility
        """
        self.generation_rate = generation_rate
        self.buffer_size = buffer_size
        
        # Thread-safe data buffer
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        
        # Streaming state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self._total_generated = 0
        self._start_time: Optional[datetime] = None
        
        # Event callbacks
        self._callbacks: List[Callable] = []
        
        # Set random seed if provided
        if random_seed:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Learn patterns from base data if provided
        self._patterns = self._extract_patterns(base_data) if base_data is not None else None
        
        # Build category/product/region weights
        self._setup_distributions()
        
        # Customer pool for realistic repeat customers
        self._customer_pool: List[str] = []
        self._max_customers = 1000
        
        logger.info(f"RealTimeDataGenerator initialized: rate={generation_rate}s, buffer={buffer_size}")
    
    def _setup_distributions(self) -> None:
        """Set up probability distributions for data generation."""
        # Category weights
        self.categories = list(CATEGORIES.keys())
        self.category_weights = [CATEGORIES[c]["weight"] for c in self.categories]
        total = sum(self.category_weights)
        self.category_weights = [w / total for w in self.category_weights]
        
        # Region weights
        self.regions = list(REGIONS.keys())
        self.region_weights = [REGIONS[r]["weight"] for r in self.regions]
        total = sum(self.region_weights)
        self.region_weights = [w / total for w in self.region_weights]
    
    def _extract_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract patterns from historical data.
        
        Args:
            df: Historical sales DataFrame
        
        Returns:
            Dictionary of learned patterns
        """
        patterns = {}
        
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Hourly patterns
            hourly = df.groupby(df["timestamp"].dt.hour)["total_amount"].mean()
            patterns["hourly"] = (hourly / hourly.mean()).to_dict()
            
            # Daily patterns
            daily = df.groupby(df["timestamp"].dt.dayofweek)["total_amount"].mean()
            patterns["daily"] = (daily / daily.mean()).to_dict()
            
            # Category preferences
            cat_counts = df["category"].value_counts(normalize=True)
            patterns["categories"] = cat_counts.to_dict()
            
            # Average order value
            patterns["avg_order_value"] = df["total_amount"].mean()
            patterns["std_order_value"] = df["total_amount"].std()
            
            logger.info("Extracted patterns from historical data")
        except Exception as e:
            logger.warning(f"Could not extract patterns: {e}")
            patterns = None
        
        return patterns
    
    def _get_time_multiplier(self) -> float:
        """
        Get sales volume multiplier based on current time.
        
        Returns:
            Multiplier for transaction probability
        """
        now = datetime.now()
        
        hour_mult = HOURLY_PATTERNS.get(now.hour, 1.0)
        day_mult = DAILY_PATTERNS.get(now.weekday(), 1.0)
        month_mult = MONTHLY_PATTERNS.get(now.month, 1.0)
        
        return hour_mult * day_mult * month_mult
    
    def _get_or_create_customer(self) -> str:
        """Get existing customer or create new one."""
        # 70% chance to reuse existing customer
        if self._customer_pool and random.random() < 0.7:
            return random.choice(self._customer_pool)
        
        new_customer = generate_customer_id()
        if len(self._customer_pool) < self._max_customers:
            self._customer_pool.append(new_customer)
        
        return new_customer
    
    def generate_transaction(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a single realistic transaction.
        
        Args:
            timestamp: Optional specific timestamp (defaults to now)
        
        Returns:
            Transaction dictionary
        """
        ts = timestamp or datetime.now()
        
        # Determine if this should be an anomaly (2% chance)
        is_anomaly = random.random() < STREAMING_CONFIG.get("anomaly_probability", 0.02)
        
        # Select category
        if self._patterns and "categories" in self._patterns:
            cats = list(self._patterns["categories"].keys())
            weights = list(self._patterns["categories"].values())
            category = np.random.choice(cats, p=weights)
        else:
            category = np.random.choice(self.categories, p=self.category_weights)
        
        # Select product from category
        products = PRODUCTS.get(category, ["Generic Product"])
        product_name = random.choice(products)
        product_id = f"PROD-{category[:3].upper()}-{products.index(product_name) + 1:04d}"
        
        # Select region
        region = np.random.choice(self.regions, p=self.region_weights)
        
        # Generate quantity
        if is_anomaly and random.random() < 0.5:
            quantity = random.randint(10, 50)  # Anomaly: high quantity
        else:
            quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.35, 0.15, 0.1, 0.15, 0.1, 0.08, 0.05, 0.02])
        
        # Generate price
        cat_info = CATEGORIES.get(category, {"avg_price": 50, "price_std": 20})
        avg_price = cat_info["avg_price"]
        price_std = cat_info["price_std"]
        
        if is_anomaly and random.random() < 0.5:
            # Anomaly: unusual price
            unit_price = avg_price * random.choice([0.2, 0.3, 2.5, 3.0])
        else:
            unit_price = max(0.99, np.random.normal(avg_price, price_std * 0.3))
        
        unit_price = round_currency(unit_price)
        total_amount = round_currency(quantity * unit_price)
        
        # Generate other fields
        customer_id = self._get_or_create_customer()
        payment_method = random.choice(PAYMENT_METHODS)
        
        transaction = {
            "timestamp": ts.isoformat(),
            "transaction_id": generate_transaction_id(),
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "quantity": int(quantity),
            "unit_price": unit_price,
            "total_amount": total_amount,
            "region": region,
            "customer_id": customer_id,
            "payment_method": payment_method,
            "is_anomaly": is_anomaly,
        }
        
        return transaction
    
    def _generation_loop(self) -> None:
        """Background thread loop for continuous data generation."""
        logger.info("Starting data generation loop")
        
        while self._running:
            try:
                # Check if we should generate based on time patterns
                multiplier = self._get_time_multiplier()
                
                # Generate transaction with probability based on time
                if random.random() < multiplier:
                    transaction = self.generate_transaction()
                    
                    with self._lock:
                        self._buffer.append(transaction)
                        self._total_generated += 1
                    
                    # Trigger callbacks
                    for callback in self._callbacks:
                        try:
                            callback(transaction)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                # Sleep for generation interval
                time.sleep(self.generation_rate)
                
            except Exception as e:
                logger.error(f"Generation loop error: {e}")
                time.sleep(1)
        
        logger.info("Data generation loop stopped")
    
    def start_streaming(self) -> None:
        """Start the background data generation thread."""
        if self._running:
            logger.warning("Streaming already running")
            return
        
        self._running = True
        self._start_time = datetime.now()
        self._thread = threading.Thread(target=self._generation_loop, daemon=True)
        self._thread.start()
        
        logger.info("Streaming started")
    
    def stop_streaming(self) -> None:
        """Stop the background data generation."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        logger.info("Streaming stopped")
    
    def add_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Add a callback to be called on each new transaction.
        
        Args:
            callback: Function that takes a transaction dict
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_latest_data(self, n: int = 100) -> pd.DataFrame:
        """
        Get the most recent n transactions.
        
        Args:
            n: Number of transactions to return
        
        Returns:
            DataFrame of recent transactions
        """
        with self._lock:
            data = list(self._buffer)[-n:]
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    def get_all_data(self) -> pd.DataFrame:
        """
        Get all data currently in buffer.
        
        Returns:
            DataFrame of all buffered transactions
        """
        with self._lock:
            data = list(self._buffer)
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    def get_data_since(self, since: datetime) -> pd.DataFrame:
        """
        Get all transactions since a given timestamp.
        
        Args:
            since: Start timestamp
        
        Returns:
            DataFrame of transactions since timestamp
        """
        df = self.get_all_data()
        
        if df.empty:
            return df
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df[df["timestamp"] >= since].reset_index(drop=True)
    
    def add_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Manually add a transaction to the buffer.
        
        Args:
            transaction: Transaction dictionary
        """
        with self._lock:
            self._buffer.append(transaction)
            self._total_generated += 1
    
    def add_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """
        Add multiple transactions to the buffer.
        
        Args:
            transactions: List of transaction dictionaries
        """
        with self._lock:
            for t in transactions:
                self._buffer.append(t)
            self._total_generated += len(transactions)
    
    def load_historical_data(self, filepath: str) -> None:
        """
        Load historical data from CSV file into buffer.
        
        Args:
            filepath: Path to CSV file
        """
        try:
            df = pd.read_csv(filepath)
            records = df.to_dict("records")
            
            # Only load most recent data up to buffer size
            records = records[-self.buffer_size:]
            
            with self._lock:
                self._buffer.clear()
                for record in records:
                    self._buffer.append(record)
            
            logger.info(f"Loaded {len(records)} historical records")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get streaming statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            buffer_size = len(self._buffer)
        
        uptime = None
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "running": self._running,
            "buffer_size": buffer_size,
            "max_buffer_size": self.buffer_size,
            "total_generated": self._total_generated,
            "generation_rate_seconds": self.generation_rate,
            "uptime_seconds": uptime,
            "transactions_per_minute": (
                self._total_generated / (uptime / 60) if uptime and uptime > 0 else 0
            ),
        }
    
    def clear_buffer(self) -> None:
        """Clear all data from the buffer."""
        with self._lock:
            self._buffer.clear()
        logger.info("Buffer cleared")


class DataIngestionService:
    """
    High-level service for managing data ingestion.
    
    Combines historical data loading and real-time generation.
    """
    
    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        enable_streaming: bool = True,
        streaming_rate: int = 5
    ):
        """
        Initialize the data ingestion service.
        
        Args:
            historical_data_path: Path to historical CSV file
            enable_streaming: Whether to enable real-time streaming
            streaming_rate: Seconds between new transactions
        """
        self.historical_data_path = historical_data_path
        self.enable_streaming = enable_streaming
        
        # Load historical data
        self.historical_data: Optional[pd.DataFrame] = None
        if historical_data_path:
            self._load_historical()
        
        # Initialize generator
        self.generator = RealTimeDataGenerator(
            base_data=self.historical_data,
            generation_rate=streaming_rate
        )
        
        # Pre-load historical data into buffer
        if self.historical_data is not None:
            records = self.historical_data.tail(5000).to_dict("records")
            self.generator.add_transactions(records)
    
    def _load_historical(self) -> None:
        """Load historical data from file."""
        try:
            self.historical_data = pd.read_csv(self.historical_data_path)
            logger.info(f"Loaded {len(self.historical_data)} historical records")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.historical_data = None
    
    def start(self) -> None:
        """Start the data ingestion service."""
        if self.enable_streaming:
            self.generator.start_streaming()
        logger.info("Data ingestion service started")
    
    def stop(self) -> None:
        """Stop the data ingestion service."""
        self.generator.stop_streaming()
        logger.info("Data ingestion service stopped")
    
    def get_data(
        self,
        include_historical: bool = True,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get combined historical and real-time data.
        
        Args:
            include_historical: Whether to include historical data
            limit: Optional limit on number of records
        
        Returns:
            Combined DataFrame
        """
        # Get real-time data
        realtime = self.generator.get_all_data()
        
        if include_historical and self.historical_data is not None:
            # Combine and deduplicate
            combined = pd.concat([self.historical_data, realtime], ignore_index=True)
            combined = combined.drop_duplicates(subset=["transaction_id"], keep="last")
        else:
            combined = realtime
        
        # Sort by timestamp
        if not combined.empty:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.sort_values("timestamp")
        
        # Apply limit
        if limit:
            combined = combined.tail(limit)
        
        return combined.reset_index(drop=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        gen_stats = self.generator.get_stats()
        
        historical_count = len(self.historical_data) if self.historical_data is not None else 0
        
        return {
            **gen_stats,
            "historical_records": historical_count,
            "total_available": historical_count + gen_stats["buffer_size"],
        }
