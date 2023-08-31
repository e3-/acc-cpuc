# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pathlib
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo
from loguru import logger
import os


# # Read Data

def read_resource_data(*, filepath: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.loc[:, "vintage"] = pd.to_datetime(df.loc[:, "vintage"], format="%Y").dt.year.values
    df.loc[:, "year"] = pd.to_datetime(df.loc[:, "year"], format="%Y").dt.year.values
    df = df.set_index(["resource", "vintage", "year"])
    return df

# ## Interpolate Costs


def _interpolate_resource_vintage(df: pd.DataFrame) -> pd.DataFrame:
    resource, vintage = df.name

    lifetimes = df["financing_lifetime"].dropna().astype(int).droplevel(["vintage", "year"])
    lifetimes = lifetimes.reset_index().drop_duplicates(subset="resource").set_index("resource")

    df = df.reset_index(["resource", "vintage"], drop=True)
    df = df.reindex(list(range(vintage, vintage + lifetimes.loc[resource].squeeze())))
    df = df.interpolate(method="linear")

    return df


# ## Discount Factors


def add_discount_factors(*, df: pd.DataFrame, discount_rate: float = 0.05, dollar_year: int = 2018) -> pd.DataFrame:
    discount_factors = (
        pd.Series(index=list(range(2022, 2100)), data=list(range(2022, 2100)))
        .rename_axis(index="year")
        .rename("discount_factor")
    )
    discount_factors = (1 + discount_rate) ** (-1 * (discount_factors - dollar_year))

    logger.info("Saving annual data with discount factors")

    return df.join(discount_factors, how="left", on="year")


# # Model


class AvoidedCostModel(pyo.ConcreteModel):
    def __init__(
        self,
        *,
        input_data: pd.DataFrame,
        price_bounds: None | pd.DataFrame = None,
        end_year: int = 2050,
        first_years_profitable: float = 1,
        interpolate_non_modeled_year_prices: bool = False,
    ):
        super().__init__()

        self.price_bounds = price_bounds
        self.input_data = input_data.sort_index()
        self.end_year = end_year
        self.prices: pd.DataFrame = pd.DataFrame()
        self.resource_cash_flow: pd.DataFrame = pd.DataFrame()

        # Sets
        self.RESOURCE_VINTAGES = pyo.Set(
            dimen=2, initialize=sorted(list({(resource, vintage) for resource, vintage, modeled_year in df.index}))
        )
        self.YEARS = pyo.Set(initialize=sorted(list(input_data.index.get_level_values(2).unique())))

        # Decision Variables
        # GHG avoided costs
        self.p_ghg = pyo.Var(
            self.YEARS,
            within=pyo.NonNegativeReals,
            initialize=lambda m, year: price_bounds.loc[year, pd.IndexSlice["p_ghg", "value"]]
            if price_bounds is not None
            else 0.0,
            bounds=lambda m, year: tuple(price_bounds.loc[year, pd.IndexSlice["p_ghg", ["lb", "ub"]]].values)
            if price_bounds is not None
            else (0, float("+inf")),
        )
        self.p_ra = pyo.Var(
            self.YEARS,
            within=pyo.NonNegativeReals,
            initialize=lambda m, year: price_bounds.loc[year, pd.IndexSlice["p_ra", "value"]]
            if price_bounds is not None
            else 0.0,
            bounds=lambda m, year: tuple(price_bounds.loc[year, pd.IndexSlice["p_ra", ["lb", "ub"]]].values)
            if price_bounds is not None
            else (0, float("+inf")),
        )

        # Calcualte each revenue 
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_ra_revenue_in_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return model.p_ra[year] * self.input_data.loc[(resource, vintage, year), "ra_capacity_mw"] * 1000

        # If emitting resource, then emission penalty
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_ghg_cost_in_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return -model.p_ghg[year] * self.input_data.loc[(resource, vintage, year), "emissions_units_per_year"]

        # Calcualte GHG contribution of resource
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_ghg_energy_revenue_impact_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return (
                model.p_ghg[year]
                * self.input_data.loc[(resource, vintage, year), "emissions_weighted_generation_tonnes_per_year"]
            )

        # Calcualte total revenue payment per year
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_payments_in_year(model, resource, vintage, year):
            return (
                model.resource_ra_revenue_in_year[resource, vintage, year]
                + model.resource_ghg_cost_in_year[resource, vintage, year]
                + model.resource_ghg_energy_revenue_impact_year[resource, vintage, year]
            )

        # Calcualte net cost per year
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_net_costs_in_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return -(
                self.input_data.loc[(resource, vintage, year), "total_annualized_fixed_costs_dollars_per_yr"]
                - self.input_data.loc[(resource, vintage, year), "net_energy_as_revenue_$_per_year"]
            )

        # Calcualte net revenue per year
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_net_revenue(model, resource, vintage, year):
            return (
                model.resource_payments_in_year[resource, vintage, year]
                + model.resource_net_costs_in_year[resource, vintage, year]
            )

        # Calculate NPV net revenue
        @self.Expression(self.RESOURCE_VINTAGES)
        def npv_net_revenue(model, resource, vintage):
            return sum(
                self.input_data.loc[(resource, vintage, year), "discount_factor"]
                * model.resource_payments_in_year[resource, vintage, year]
                if year in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index
                else 0.0
                for year in model.YEARS
            ) + sum(
                self.input_data.loc[(resource, vintage, year), "discount_factor"]
                * model.resource_net_costs_in_year[resource, vintage, year]
                if year in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index
                else 0.0
                for year in model.YEARS
            )
        
        # Calculate first year(s) NPV net revenue
        @self.Expression(self.RESOURCE_VINTAGES)
        def first_year_net_revenue(model, resource, vintage):
            return sum(
                self.input_data.loc[(resource, vintage, year), "discount_factor"]
                * model.resource_payments_in_year[resource, vintage, year]
                if year in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index and (year <= vintage + first_years_profitable - 1)
                else 0.0
                for year in model.YEARS
            ) + sum(
                self.input_data.loc[(resource, vintage, year), "discount_factor"]
                * model.resource_net_costs_in_year[resource, vintage, year]
                if year in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index and (year <= vintage + first_years_profitable - 1)
                else 0.0
                for year in model.YEARS
            )

        # Constraints
        # NPV net revenues >= 0 
        @self.Constraint(self.RESOURCE_VINTAGES)
        def constrain_npv_net_revenue(model, resource, vintage):
            return model.npv_net_revenue[resource, vintage] >= 0.0

        # NPV first year(s) net revenues >= 0
        @self.Constraint(self.RESOURCE_VINTAGES)
        def constrain_first_year_net_revenue(model, resource, vintage):
            return model.first_year_net_revenue[resource, vintage] >= 0.0

        # GHG avoided costs constant after 2045
        @self.Constraint(self.YEARS)
        def constant_p_ghg_after_2045(model, year):
            if year > 2045:
                return model.p_ghg[year] == model.p_ghg[year - 1]
            return pyo.Constraint.Skip

        # Capacity avoided costs constant after 2045
        @self.Constraint(self.YEARS)
        def constant_p_ra_after_2045(model, year):
            if year > 2045:
                return model.p_ra[year] == model.p_ra[year - 1]
            return pyo.Constraint.Skip

        ## Objectives
        @self.Objective(sense=pyo.minimize)
        def npv_net_payments(model):
            return sum(model.npv_net_revenue[resource, vintage] for resource, vintage in model.RESOURCE_VINTAGES)

    def update_results_attributes(self):
        self.prices = pd.DataFrame.from_dict(
            {
                col: {year: pyo.value(p[year]) for year in self.YEARS}
                for col, p in {
                    "GHG": self.p_ghg,
                    "Gen Capacity": self.p_ra,
                }.items()
            }
        )

    def solve(self, tee: bool = False):
        if sys.platform=="win32":
            opt = pyo.SolverFactory(
                "cbc",
                solver_io="lp",
                executable="../solvers/cbc.exe",
            )
        else:
            opt = pyo.SolverFactory(
                "cbc",
                solver_io="lp",
            )
        solution = opt.solve(self, tee=tee)

        if solution.solver.termination_condition == pyo.TerminationCondition.optimal:
            logger.info(f"Objective function value (NPV net payments): ${pyo.value(self.npv_net_payments):,.2f}")

            self.update_results_attributes()

        elif solution.solver.termination_condition == pyo.TerminationCondition.infeasible:
            logger.critical("Model infeasible.")

    # post process results after getting avoided cost prices
    def plot_avoided_costs(self):
        fig = px.line(
            m.prices,
            title="Annual Avoided Costs ($2018)"
        )
        
        fig.update_layout(template="plotly_white", height=400, width=600)
        fig.update_xaxes(title_text="Year")
        
        return fig

    # Calculate cashflow for one resource and one vintage
    def calculate_one_resource_cash_flow(self, *, resource_vintage: tuple):
        resource, vintage = resource_vintage

        resource_cash_flow = pd.DataFrame.from_dict(
            {
                col: {year: pyo.value(p[resource, vintage, year]) for year in self.YEARS}
                for col, p in {
                    "Gen Cap Value": self.resource_ra_revenue_in_year,
                    "GHG Cost": self.resource_ghg_cost_in_year,
                    "GHG Value": self.resource_ghg_energy_revenue_impact_year,
                }.items()
            }
        )

        resource_inputs = (
            self.input_data.xs((resource, vintage), level=("resource", "vintage"))
            .loc[:, ["total_annualized_fixed_costs_dollars_per_yr", "net_energy_as_revenue_$_per_year"]]
            .rename(
                columns={
                    "total_annualized_fixed_costs_dollars_per_yr": "Fixed Cost",
                    "net_energy_as_revenue_$_per_year": "Net Energy + AS Revenue",
                }
            )
        )

        resource_cash_flow = pd.concat(
            [
                resource_inputs,
                resource_cash_flow.reindex(resource_inputs.index),
            ],
            axis=1,
        ).sort_index()

        return resource_cash_flow
    
    # Calculate cashflow for all reousrces and all vintages
    def calculate_all_resource_cashflow(self):
        all_resource_cashflow = pd.DataFrame()
        
        for resource, vintage in m.RESOURCE_VINTAGES:
            cashflow = self.calculate_one_resource_cash_flow(resource_vintage=(resource, vintage))

            # Create a DataFrame for the current vintage's cash flow data
            cashflow_df = pd.DataFrame(cashflow, columns=cashflow.columns.values)

            # Add 'Resource' and 'Vintage' columns to the DataFrame and set their values
            cashflow_df["Resource"] = resource
            cashflow_df["Vintage"] = vintage

            # Concatenate the current vintage's DataFrame with the main DataFrame
            all_resource_cashflow = pd.concat([all_resource_cashflow, cashflow_df])
        
        return all_resource_cashflow
    
    # Calculate NPV of costs and revenues for all resources
    def calculate_all_resource_npv(self):
        discount_factors = (
            self.input_data.loc[:, "discount_factor"].reset_index(["resource", "vintage"], drop=True).drop_duplicates()
        )

        npv_results = {}
        for resource, vintage in self.RESOURCE_VINTAGES:
            cashflow = self.calculate_one_resource_cash_flow(resource_vintage=(resource, vintage))
            npv = cashflow.mul(discount_factors.reindex(cashflow.index), axis=0).sum()
            npv_results[(resource, vintage)] = npv

        all_resource_npv = pd.concat(npv_results, axis=1).T
        all_resource_npv.index.names = ["resource", "vintage"]
        return all_resource_npv

    # Plot cashflow for one resource and one vintage
    def plot_resource_cash_flow(self, *, resource_vintage: tuple):
        resource, vintage = resource_vintage
        cash_flow = self.calculate_one_resource_cash_flow(resource_vintage=(resource, vintage))

        fig = px.bar(
            cash_flow,
            x=cash_flow.index,
            y=cash_flow.columns[1:],
            title="{} Vintage {} Cash Flow ($/kW)".format(resource, vintage),
            # color_discrete_sequence = ['orange', 'white', 'green', 'white', 'purple', 'red'],
            barmode="stack",
        )

        fig.add_scatter(
            x=cash_flow.index,
            y=cash_flow["Fixed Cost"],
            mode="markers",
            showlegend=False,
            marker=dict(color="#808080"),
            row=1,
            col=1,
        )
        fig.update_layout(template="plotly_white", height=400, width=1000)

        return fig

    # Plot NPV of costs and revenues for one resource
    def plot_npv_net_revenues(self, *, resource):
        resource_npv = self.calculate_all_resource_npv().loc[resource]

        fig = px.bar(
            resource_npv,
            x=resource_npv.index,
            y=resource_npv.columns[1:],
            title="{} NPV by Vintage ($/kW)".format(resource),
            barmode="stack",
        )

        fig.add_scatter(
            x=resource_npv.index,
            y=resource_npv["Fixed Cost"],
            mode="markers",
            showlegend=False,
            marker=dict(color="#808080"),
            row=1,
            col=1,
        )
        fig.update_layout(template="plotly_white", height=400, width=1000)

        return fig


# # Running Model

# +
# User input
run_id = "07012023_staff_proposal"
base_path = pathlib.Path.cwd()


# Create output folder
outputs_file_path = base_path / "results" / run_id
outputs_file_path.mkdir(exist_ok=True, parents=True)

# Read resource data csv
df = read_resource_data(filepath=base_path / "data" / "processed" / run_id / "resource_data.csv")
# df = df.groupby(["resource", "vintage"]).apply(_interpolate_resource_vintage)
df = add_discount_factors(df=df, discount_rate=0.054, dollar_year=2018)

# Run model
m = AvoidedCostModel(
    input_data=df, 
    first_years_profitable=1, 
    price_bounds=pd.read_csv(base_path / "data" / "processed" / run_id / "price_bounds.csv", index_col=0, header=[0, 1])
)

m.solve()

# Output results
m.prices.to_csv(outputs_file_path / "avoided_costs.csv")
m.calculate_all_resource_cashflow().to_csv(outputs_file_path / "cash_flow_by_resource.csv")
m.calculate_all_resource_npv().to_csv(outputs_file_path / "npv_by_resource.csv")
# -

# # Plot Results

# Plot avoided costs
m.plot_avoided_costs()

# Plot cash flow for one example resource and vintage
m.plot_resource_cash_flow(resource_vintage=("generic_solar", 2030))

# Plot NPV for one example resource
m.plot_npv_net_revenues(resource="generic_solar")


