import pandas as pd
import numpy as np
import io
import traceback
from pydantic import BaseModel
from typing import Optional
import copy

# ─── Pydantic Models ───────────────────────────────────────────────

class Observation(BaseModel):
    data_preview: str
    error_message: str
    schema_info: str
    step_number: int
    task_name: str

class Action(BaseModel):
    code: str

class Reward(BaseModel):
    value: float
    reason: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

# ─── Task Definitions ──────────────────────────────────────────────

def get_easy_task():
    """Easy: Fix CSV with wrong delimiter, encoding, OCR errors — 50 rows"""
    import random
    random.seed(7)

    names_clean = [
        'Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Grace', 'Henry',
        'Iris', 'Jack', 'Karen', 'Leo', 'Mia', 'Nate', 'Olivia', 'Paul',
        'Quinn', 'Rachel', 'Sam', 'Tina', 'Uma', 'Victor', 'Wendy', 'Xander',
        'Yara', 'Zoe', 'Aaron', 'Bella', 'Carlos', 'Diana', 'Ethan', 'Fiona',
        'George', 'Hannah', 'Ivan', 'Julia', 'Kevin', 'Laura', 'Mike', 'Nancy',
        'Oscar', 'Penny', 'Raj', 'Sara', 'Tom', 'Uma', 'Vera', 'Will', 'Xena', 'Yusuf'
    ]
    ages_clean   = [25,0,35,40,28,32,27,45,31,38,22,55,29,41,36,48,26,33,44,30,
                    37,52,23,39,28,34,46,21,43,27,50,35,29,38,31,42,24,36,49,26,
                    33,47,22,41,30,38,25,44,32,28]
    salaries_clean = [50000,60000,70000,80000,55000,62000,48000,90000,67000,75000,
                      42000,110000,58000,85000,72000,95000,46000,66000,88000,54000,
                      71000,105000,44000,78000,53000,63000,92000,40000,84000,51000,
                      100000,69000,57000,76000,64000,83000,45000,70000,97000,49000,
                      65000,93000,43000,82000,56000,74000,48000,87000,61000,52000]
    domains = ['test.com','example.com','mail.com','work.org','uni.edu']

    # Build broken data by injecting errors
    broken_names = []
    broken_ages  = []
    broken_sals  = []
    broken_emails= []

    for i, name in enumerate(names_clean):
        # inject semicolons into every 3rd name
        bn = (name[:1] + ';' + name[1:]) if i % 3 == 1 else name
        broken_names.append(bn)

        age = ages_clean[i]
        # inject 'abc' for 0-age rows, else OCR 'O' for 4→'4O' pattern every 5th
        if age == 0:
            broken_ages.append('abc')
        elif i % 5 == 3:
            broken_ages.append(str(age).replace('4','4O').replace('8','8O') if '4' in str(age) or '8' in str(age) else str(age))
        else:
            broken_ages.append(str(age))

        sal = salaries_clean[i]
        sal_str = str(sal)
        # inject semicolon every 4th, OCR every 7th
        if i % 4 == 1:
            sal_str = sal_str[:2] + ';' + sal_str[2:]
        if i % 7 == 0 and '8' in sal_str:
            sal_str = sal_str.replace('8','8O',1)
        broken_sals.append(sal_str)

        domain = domains[i % len(domains)]
        if i % 6 == 1:
            broken_emails.append(f"{name.lower()}@")       # missing domain
        elif i % 6 == 4:
            broken_emails.append(f"{name.lower()}nodomain") # no @ at all
        else:
            broken_emails.append(f"{name.lower()}@{domain}")

    broken_data = pd.DataFrame({
        'name':   broken_names,
        'age':    broken_ages,
        'salary': broken_sals,
        'email':  broken_emails,
    })

    # Build expected data
    fixed_emails = []
    for i, name in enumerate(names_clean):
        domain = domains[i % len(domains)]
        if i % 6 == 1:
            fixed_emails.append(f"{name.lower()}@{domain}")
        elif i % 6 == 4:
            fixed_emails.append(f"{name.lower()}nodomain@test.com")
        else:
            fixed_emails.append(f"{name.lower()}@{domain}")

    expected_data = pd.DataFrame({
        'name':   names_clean,
        'age':    ages_clean,
        'salary': salaries_clean,
        'email':  fixed_emails,
    })
    return broken_data, expected_data


def get_medium_task():
    """Medium: Fix schema errors, type mismatches, null handling — 80 rows"""
    np.random.seed(99)
    n = 80

    # product_id: inject Nones at every 8th row
    product_ids_clean = list(range(1, n + 1))
    product_ids_broken = [None if i % 8 == 0 else float(v) for i, v in enumerate(product_ids_clean)]

    # price: inject broken values
    prices_clean = np.round(np.random.uniform(1.0, 200.0, n), 2).tolist()
    broken_prices = []
    for i, p in enumerate(prices_clean):
        if i % 9 == 0:
            broken_prices.append('N/A')
        elif i % 9 == 4:
            broken_prices.append('free')
        elif i % 9 == 7:
            broken_prices.append(str(-abs(p)))  # negative price
        else:
            broken_prices.append(str(round(p, 2)))

    prices_expected = []
    for i, p in enumerate(prices_clean):
        if i % 9 in (0, 4):
            prices_expected.append(0.0)
        elif i % 9 == 7:
            prices_expected.append(0.0)
        else:
            prices_expected.append(round(p, 2))

    # quantity: inject Nones
    quantities_clean = np.random.randint(1, 200, n).tolist()
    quantities_broken = [None if i % 7 == 0 else q for i, q in enumerate(quantities_clean)]
    quantities_expected = [0 if i % 7 == 0 else q for i, q in enumerate(quantities_clean)]

    # category: inject Nones
    cats = ['Electronics', 'Clothing', 'Food', 'Books', 'Sports']
    categories_clean = [cats[i % len(cats)] for i in range(n)]
    categories_broken = [None if i % 11 == 0 else c for i, c in enumerate(categories_clean)]
    categories_expected = ['Unknown' if i % 11 == 0 else c for i, c in enumerate(categories_clean)]

    # in_stock: mix string representations
    stock_map_broken = ['yes','no','yes','true','false','YES','NO','True','False','1']
    stock_map_expected = [True,False,True,True,False,True,False,True,False,True]
    in_stock_broken   = [stock_map_broken[i % 10] for i in range(n)]
    in_stock_expected = [stock_map_expected[i % 10] for i in range(n)]

    broken_data = pd.DataFrame({
        'product_id': product_ids_broken,
        'price':      broken_prices,
        'quantity':   quantities_broken,
        'category':   categories_broken,
        'in_stock':   in_stock_broken,
    })
    expected_data = pd.DataFrame({
        'product_id': product_ids_clean,
        'price':      prices_expected,
        'quantity':   quantities_expected,
        'category':   categories_expected,
        'in_stock':   in_stock_expected,
    })
    return broken_data, expected_data


def get_hard_task():
    """Hard: Fix logic bugs in aggregation pipeline — 200 rows, 5 regions"""
    np.random.seed(42)
    n = 200
    regions = ['North', 'South', 'East', 'West', 'Central']
    region_col = [regions[i % len(regions)] for i in range(n)]

    sales_raw = np.round(np.random.uniform(100, 1000, n), 2)
    # Inject None into ~15% of sales
    sales_col = [None if np.random.random() < 0.15 else float(s) for s in sales_raw]
    sales_filled = np.array([0.0 if s is None else s for s in sales_col])

    returns_col = np.round(np.random.uniform(5, 100, n), 2).tolist()

    discount_pcts = np.random.choice([0, 5, 10, 15, 20, 25], n)
    discount_str  = [f"{d}%" for d in discount_pcts]
    discount_frac = discount_pcts / 100.0

    net_sales = sales_filled * (1 - discount_frac) - np.array(returns_col)

    broken_data = pd.DataFrame({
        'date':     pd.date_range('2024-01-01', periods=n, freq='D').astype(str),
        'region':   region_col,
        'sales':    sales_col,
        'returns':  returns_col,
        'discount': discount_str,
    })

    # Compute expected aggregation
    temp = pd.DataFrame({
        'region':    region_col,
        'net_sales': net_sales,
        'discount':  discount_frac,
        'sales':     sales_filled,
    })
    expected_data = temp.groupby('region').agg(
        total_net_sales=('net_sales', 'sum'),
        avg_discount=('discount', 'mean'),
        num_transactions=('sales', 'count')
    ).reset_index().sort_values('region').reset_index(drop=True)
    expected_data['total_net_sales'] = expected_data['total_net_sales'].round(2)
    expected_data['avg_discount']    = expected_data['avg_discount'].round(6)

    return broken_data, expected_data



# ─── Main Environment Class ────────────────────────────────────────

class DataPipelineEnv:
    TASKS = ['fix_csv_encoding', 'fix_schema_errors', 'optimize_pipeline']

    def __init__(self):
        self.current_task = 'fix_csv_encoding'
        self.current_df = None
        self.expected_df = None
        self.step_num = 0
        self.max_steps = 20
        self.done = False
        self.error_message = ""
        self.episode_history = []

    def reset(self, task_name: str = None) -> Observation:
        if task_name:
            self.current_task = task_name
        self.step_num = 0
        self.done = False
        self.error_message = ""
        self.episode_history = []

        if self.current_task == 'fix_csv_encoding':
            self.current_df, self.expected_df = get_easy_task()
            self.max_steps = 10
        elif self.current_task == 'fix_schema_errors':
            self.current_df, self.expected_df = get_medium_task()
            self.max_steps = 15
        elif self.current_task == 'optimize_pipeline':
            self.current_df, self.expected_df = get_hard_task()
            self.max_steps = 20

        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"message": "Episode already done. Call reset()."}
            )

        self.step_num += 1
        prev_df = copy.deepcopy(self.current_df)
        reward = 0.0
        self.error_message = ""

        try:
            local_vars = {"df": copy.deepcopy(self.current_df), "pd": pd, "np": np}
            exec(action.code, {}, local_vars)
            new_df = local_vars.get("df", self.current_df)
            self.current_df = new_df
            score = self._compute_score()
            reward = score - self._compute_score_for(prev_df)
            reward = max(-1.0, min(1.0, reward))

            if score >= 0.95:
                self.done = True
                reward = 1.0

        except Exception as e:
            self.error_message = str(e)
            reward = -0.1

        if self.step_num >= self.max_steps:
            self.done = True

        self.episode_history.append({
            "step": self.step_num,
            "action": action.code,
            "reward": reward,
            "error": self.error_message
        })

        return StepResult(
            observation=self._get_observation(),
            reward=round(reward, 4),
            done=self.done,
            info={
                "step": self.step_num,
                "score": round(self._compute_score(), 4),
                "error": self.error_message
            }
        )

    def state(self) -> dict:
        return {
            "task": self.current_task,
            "step": self.step_num,
            "max_steps": self.max_steps,
            "done": self.done,
            "score": round(self._compute_score(), 4),
            "data_preview": self.current_df.head().to_string() if self.current_df is not None else "",
            "episode_history": self.episode_history
        }

    def _get_observation(self) -> Observation:
        if self.current_df is None:
            return Observation(
                data_preview="No data",
                error_message=self.error_message,
                schema_info="No schema",
                step_number=self.step_num,
                task_name=self.current_task
            )
        schema = []
        for col in self.current_df.columns:
            null_count = self.current_df[col].isnull().sum()
            schema.append(f"{col}: {self.current_df[col].dtype} (nulls: {null_count})")

        return Observation(
            data_preview=self.current_df.head(5).to_string(),
            error_message=self.error_message,
            schema_info="\n".join(schema),
            step_number=self.step_num,
            task_name=self.current_task
        )

    def _compute_score(self) -> float:
        return self._compute_score_for(self.current_df)

    def _compute_score_for(self, df) -> float:
        if df is None or self.expected_df is None:
            return 0.0
        try:
            score = 0.0
            total_cols = len(self.expected_df.columns)
            for col in self.expected_df.columns:
                if col not in df.columns:
                    continue
                try:
                    expected_col = self.expected_df[col]
                    actual_col = df[col].reset_index(drop=True)
                    expected_col = expected_col.reset_index(drop=True)
                    if expected_col.dtype in [np.float64, np.int64]:
                        actual_col = pd.to_numeric(actual_col, errors='coerce').fillna(0)
                        expected_col = pd.to_numeric(expected_col, errors='coerce').fillna(0)
                        matches = np.isclose(actual_col, expected_col, atol=1.0).sum()
                    else:
                        matches = (actual_col.astype(str) == expected_col.astype(str)).sum()
                    score += matches / len(expected_col)
                except Exception:
                    continue
            return round(score / total_cols, 4)
        except Exception:
            return 0.0