# data/financebench_loader.py
import pandas as pd

class FinanceBenchLoader:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_parquet(dataset_path)
        
    def get_earnings_report(self, company: str, year: int):
        """Retrieve structured earnings data"""
        return self.dataset[
            (self.dataset['company'] == company) & 
            (self.dataset['year'] == year)
        ].to_dict('records')