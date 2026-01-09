"""
Synthetic Sales Data Generator.
Creates realistic sales transaction data with configurable patterns for testing and demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random
import argparse

from config import (
    DATASET_CONFIG, CATEGORIES, PRODUCTS, REGIONS, PAYMENT_METHODS,
    HOURLY_PATTERNS, DAILY_PATTERNS, MONTHLY_PATTERNS,
    RAW_DATA_DIR, PROCESSED_DATA_DIR
)
from utils import (
    generate_transaction_id, generate_customer_id, generate_product_id,
    setup_logger, round_currency
)

logger = setup_logger("data_generator")


class SalesDataGenerator:
    """
    Generates synthetic sales transaction data with realistic patterns.
    
    Attributes:
        num_transactions: Number of transactions to generate
        start_date: Start date for data generation
        end_date: End date for data generation
        anomaly_rate: Proportion of anomalous transactions
    """
    
    def __init__(
        self,
        num_transactions: int = 15000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        anomaly_rate: float = 0.02,
        random_seed: int = 42
    ):
        """
        Initialize the data generator.
        
        Args:
            num_transactions: Number of transactions to generate
            start_date: Start date (defaults to 365 days ago)
            end_date: End date (defaults to now)
            anomaly_rate: Proportion of anomalous transactions (0-1)
            random_seed: Random seed for reproducibility
        """
        self.num_transactions = num_transactions
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=365))
        self.anomaly_rate = anomaly_rate
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Build category and product mappings
        self._build_category_weights()
        self._build_product_catalog()
        self._build_region_weights()
        
        # Track generated customer IDs for realistic distribution
        self.customer_pool: List[str] = []
        self.customer_pool_size = min(num_transactions // 5, 5000)
        
        logger.info(f"Initialized generator for {num_transactions} transactions")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
    
    def _build_category_weights(self) -> None:
        """Build category selection weights."""
        self.categories = list(CATEGORIES.keys())
        self.category_weights = [CATEGORIES[c]["weight"] for c in self.categories]
        # Normalize weights
        total = sum(self.category_weights)
        self.category_weights = [w / total for w in self.category_weights]
    
    def _build_product_catalog(self) -> None:
        """Build product catalog with IDs."""
        self.product_catalog: Dict[str, List[Dict]] = {}
        
        for category, products in PRODUCTS.items():
            self.product_catalog[category] = []
            for i, product_name in enumerate(products):
                self.product_catalog[category].append({
                    "product_id": generate_product_id(category, i + 1),
                    "product_name": product_name,
                    "category": category,
                })
    
    def _build_region_weights(self) -> None:
        """Build region selection weights."""
        self.regions = list(REGIONS.keys())
        self.region_weights = [REGIONS[r]["weight"] for r in self.regions]
        total = sum(self.region_weights)
        self.region_weights = [w / total for w in self.region_weights]
    
    def _get_or_create_customer(self) -> str:
        """Get an existing customer or create a new one."""
        # 70% chance to reuse existing customer if pool is populated
        if self.customer_pool and random.random() < 0.7:
            return random.choice(self.customer_pool)
        
        # Create new customer
        new_customer = generate_customer_id()
        if len(self.customer_pool) < self.customer_pool_size:
            self.customer_pool.append(new_customer)
        
        return new_customer
    
    def _generate_timestamp(self, base_date: datetime) -> datetime:
        """
        Generate a timestamp with realistic time-of-day patterns.
        
        Args:
            base_date: Date for the transaction
        
        Returns:
            Datetime with realistic hour
        """
        # Sample hour based on hourly patterns
        hours = list(HOURLY_PATTERNS.keys())
        hour_weights = list(HOURLY_PATTERNS.values())
        total = sum(hour_weights)
        hour_weights = [w / total for w in hour_weights]
        
        hour = np.random.choice(hours, p=hour_weights)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
    
    def _get_time_multiplier(self, timestamp: datetime) -> float:
        """
        Calculate sales volume multiplier based on time patterns.
        
        Args:
            timestamp: Transaction timestamp
        
        Returns:
            Multiplier for quantity/probability
        """
        hour_mult = HOURLY_PATTERNS.get(timestamp.hour, 1.0)
        day_mult = DAILY_PATTERNS.get(timestamp.weekday(), 1.0)
        month_mult = MONTHLY_PATTERNS.get(timestamp.month, 1.0)
        
        return hour_mult * day_mult * month_mult
    
    def _generate_quantity(self, category: str, is_anomaly: bool = False) -> int:
        """
        Generate realistic quantity based on category.
        
        Args:
            category: Product category
            is_anomaly: Whether this is an anomalous transaction
        
        Returns:
            Quantity ordered
        """
        # Base quantity distribution (most orders are 1-3 items)
        if is_anomaly:
            # Anomaly: unusually high or low
            if random.random() < 0.5:
                return random.randint(15, 50)  # Very high
            else:
                return 1  # Normal (anomaly in price instead)
        
        # Normal distribution centered around 1-2
        base_qty = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.35, 0.15, 0.1, 0.15, 0.1, 0.08, 0.05, 0.02])
        return max(1, int(base_qty))
    
    def _generate_price(self, category: str, is_anomaly: bool = False) -> float:
        """
        Generate realistic price based on category.
        
        Args:
            category: Product category
            is_anomaly: Whether this is an anomalous transaction
        
        Returns:
            Unit price
        """
        cat_info = CATEGORIES[category]
        avg_price = cat_info["avg_price"]
        price_std = cat_info["price_std"]
        
        if is_anomaly:
            # Anomaly: price way off from normal
            if random.random() < 0.5:
                # Very high price (premium/error)
                price = avg_price * random.uniform(2.5, 5.0)
            else:
                # Very low price (discount/error)
                price = avg_price * random.uniform(0.1, 0.3)
        else:
            # Normal price with some variation
            price = np.random.normal(avg_price, price_std * 0.5)
        
        # Ensure minimum price
        price = max(0.99, price)
        
        # Round to realistic price points
        return round_currency(price)
    
    def generate_transaction(
        self,
        timestamp: Optional[datetime] = None,
        force_anomaly: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a single sales transaction.
        
        Args:
            timestamp: Optional specific timestamp
            force_anomaly: Force this to be an anomalous transaction
        
        Returns:
            Transaction dictionary
        """
        # Determine if anomaly
        is_anomaly = force_anomaly or (random.random() < self.anomaly_rate)
        
        # Generate timestamp if not provided
        if timestamp is None:
            days_range = (self.end_date - self.start_date).days
            random_day = self.start_date + timedelta(days=random.randint(0, days_range))
            timestamp = self._generate_timestamp(random_day)
        
        # Select category and product
        category = np.random.choice(self.categories, p=self.category_weights)
        product = random.choice(self.product_catalog[category])
        
        # Select region
        region = np.random.choice(self.regions, p=self.region_weights)
        
        # Generate quantity and price
        quantity = self._generate_quantity(category, is_anomaly)
        unit_price = self._generate_price(category, is_anomaly)
        total_amount = round_currency(quantity * unit_price)
        
        # Generate customer and payment info
        customer_id = self._get_or_create_customer()
        payment_method = random.choice(PAYMENT_METHODS)
        
        # Build transaction
        transaction = {
            "timestamp": timestamp.isoformat(),
            "transaction_id": generate_transaction_id(),
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "category": category,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_amount": total_amount,
            "region": region,
            "customer_id": customer_id,
            "payment_method": payment_method,
            "is_anomaly": is_anomaly,
        }
        
        return transaction
    
    def generate_dataset(self, sort_by_time: bool = True) -> pd.DataFrame:
        """
        Generate complete sales dataset.
        
        Args:
            sort_by_time: Whether to sort by timestamp
        
        Returns:
            DataFrame with all transactions
        """
        logger.info(f"Generating {self.num_transactions} transactions...")
        
        transactions = []
        
        # Generate dates spread across the range
        days_range = (self.end_date - self.start_date).days
        
        for i in range(self.num_transactions):
            # Distribute transactions across time with some clustering
            day_offset = random.randint(0, days_range)
            base_date = self.start_date + timedelta(days=day_offset)
            
            # Apply time multiplier for realistic distribution
            time_mult = self._get_time_multiplier(base_date)
            
            # More transactions during high-traffic periods
            if random.random() > time_mult * 0.5:
                continue
            
            transaction = self.generate_transaction()
            transactions.append(transaction)
            
            if (i + 1) % 5000 == 0:
                logger.info(f"Generated {i + 1}/{self.num_transactions} transactions")
        
        # Fill remaining to reach target
        while len(transactions) < self.num_transactions:
            transaction = self.generate_transaction()
            transactions.append(transaction)
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        if sort_by_time:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} transactions")
        logger.info(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
        
        return df
    
    def inject_specific_anomalies(self, df: pd.DataFrame, anomaly_configs: List[Dict]) -> pd.DataFrame:
        """
        Inject specific anomaly patterns into the dataset.
        
        Args:
            df: Existing DataFrame
            anomaly_configs: List of anomaly configurations
        
        Returns:
            DataFrame with injected anomalies
        """
        df = df.copy()
        
        for config in anomaly_configs:
            anomaly_type = config.get("type", "spike")
            target_date = config.get("date")
            magnitude = config.get("magnitude", 2.0)
            
            if anomaly_type == "spike":
                # Revenue spike on specific date
                mask = df["timestamp"].str.startswith(target_date)
                df.loc[mask, "total_amount"] *= magnitude
                df.loc[mask, "is_anomaly"] = True
                
            elif anomaly_type == "drop":
                # Revenue drop
                mask = df["timestamp"].str.startswith(target_date)
                df.loc[mask, "total_amount"] /= magnitude
                df.loc[mask, "is_anomaly"] = True
                
            elif anomaly_type == "region_drop":
                # Specific region drops
                region = config.get("region")
                mask = (df["timestamp"].str.startswith(target_date)) & (df["region"] == region)
                df.loc[mask, "total_amount"] /= magnitude
                df.loc[mask, "is_anomaly"] = True
        
        return df
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str = "sales_data.csv",
        directory: str = None
    ) -> str:
        """
        Save dataset to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            directory: Output directory (defaults to RAW_DATA_DIR)
        
        Returns:
            Path to saved file
        """
        if directory is None:
            directory = RAW_DATA_DIR
        
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved dataset to {filepath}")
        logger.info(f"Shape: {df.shape}")
        
        return filepath
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: Sales DataFrame
        
        Returns:
            Dictionary of summary statistics
        """
        return {
            "total_transactions": len(df),
            "total_revenue": round_currency(df["total_amount"].sum()),
            "avg_order_value": round_currency(df["total_amount"].mean()),
            "date_range": {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
            },
            "categories": df["category"].nunique(),
            "products": df["product_id"].nunique(),
            "customers": df["customer_id"].nunique(),
            "regions": df["region"].nunique(),
            "anomaly_count": df["is_anomaly"].sum(),
            "category_breakdown": df.groupby("category")["total_amount"].sum().to_dict(),
            "region_breakdown": df.groupby("region")["total_amount"].sum().to_dict(),
        }


def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description="Generate synthetic sales data")
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=15000,
        help="Number of transactions to generate"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data"
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.02,
        help="Proportion of anomalous transactions"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sales_data.csv",
        help="Output filename"
    )
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create generator
    generator = SalesDataGenerator(
        num_transactions=args.num_transactions,
        start_date=start_date,
        end_date=end_date,
        anomaly_rate=args.anomaly_rate,
        random_seed=args.seed
    )
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Inject some specific anomaly patterns
    anomaly_configs = [
        {
            "type": "spike",
            "date": (end_date - timedelta(days=30)).strftime("%Y-%m-%d"),
            "magnitude": 2.5
        },
        {
            "type": "drop",
            "date": (end_date - timedelta(days=60)).strftime("%Y-%m-%d"),
            "magnitude": 3.0
        },
        {
            "type": "region_drop",
            "date": (end_date - timedelta(days=15)).strftime("%Y-%m-%d"),
            "region": "Europe",
            "magnitude": 2.0
        },
    ]
    
    df = generator.inject_specific_anomalies(df, anomaly_configs)
    
    # Save dataset
    generator.save_dataset(df, args.output)
    
    # Print summary
    stats = generator.generate_summary_stats(df)
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total Transactions: {stats['total_transactions']:,}")
    print(f"Total Revenue: ${stats['total_revenue']:,.2f}")
    print(f"Average Order Value: ${stats['avg_order_value']:.2f}")
    print(f"Date Range: {stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}")
    print(f"Unique Categories: {stats['categories']}")
    print(f"Unique Products: {stats['products']}")
    print(f"Unique Customers: {stats['customers']}")
    print(f"Regions: {stats['regions']}")
    print(f"Anomalies: {stats['anomaly_count']}")
    print("\nCategory Revenue Breakdown:")
    for cat, revenue in sorted(stats['category_breakdown'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: ${revenue:,.2f}")
    print("\nRegion Revenue Breakdown:")
    for region, revenue in sorted(stats['region_breakdown'].items(), key=lambda x: -x[1]):
        print(f"  {region}: ${revenue:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
