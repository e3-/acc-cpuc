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

import warnings
warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas."
)
import pathlib
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo
from loguru import logger
import os
from pyomo.environ import Var, Param, NonNegativeReals

# # specify parameters

## Parameters (2024ACC_TRC)
RUN_ID = "2024ACC_TRC"
FIRST_YEARS_PROFITABLE_CONSTRAINT = True # True to enable first years to be profitable constraints
FIRST_YEARS_PROFITABLE = 1 # First number of years to be profitable
DISCOUNT_RATE = 0.05196 # real discount rate converted from 7.3% of nominal WACC
DOLLAR_YEAR = 2022 

## Parameters (2024ACC_SCT_Base)
# RUN_ID = "2024ACC_SCT_Base"
# FIRST_YEARS_PROFITABLE_CONSTRAINT = True # True to enable first years to be profitable constraints
# FIRST_YEARS_PROFITABLE = 1 # First number of years to be profitable
# DISCOUNT_RATE = 0.03 # real SCT discount rate
# DOLLAR_YEAR = 2022 

## Parameters (2024ACC_SCT_High)
# RUN_ID = "2024ACC_SCT_High"
# FIRST_YEARS_PROFITABLE_CONSTRAINT = True # True to enable first years to be profitable constraints
# FIRST_YEARS_PROFITABLE = 1 # First number of years to be profitable
# DISCOUNT_RATE = 0.03 # real SCT discount rate
# DOLLAR_YEAR = 2022 

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
        self.enable_first_year_revenue_constraint = pyo.Param(mutable=True, initialize=1, within=pyo.Binary)
        
        # Define the years for which p_ghg and p_ra should be explicitly solved
        self.MODELED_YEARS = pyo.Set(initialize=[*range(2024, 2035), 2040, 2045])

        # Sets
        self.RESOURCE_VINTAGES = pyo.Set(
            dimen=2, initialize=sorted(list({(resource, vintage) for resource, vintage, modeled_year in input_data.index}))
        )
        self.YEARS = pyo.Set(initialize=sorted(list(input_data.index.get_level_values(2).unique())))

        # Decision Variables
        # GHG avoided costs
        self.p_ghg = pyo.Var(
            self.MODELED_YEARS,
            within=pyo.NonNegativeReals,
            initialize=lambda m, year: price_bounds.loc[year, pd.IndexSlice["p_ghg", "value"]]
            if price_bounds is not None
            else 0.0,
            bounds=lambda m, year: tuple(price_bounds.loc[year, pd.IndexSlice["p_ghg", ["lb", "ub"]]].values)
            if price_bounds is not None
            else (0, float("+inf")),
        )
        # RA avoided costs
        self.p_ra = pyo.Var(
            self.MODELED_YEARS,
            within=pyo.NonNegativeReals,
            initialize=lambda m, year: price_bounds.loc[year, pd.IndexSlice["p_ra", "value"]]
            if price_bounds is not None
            else 0.0,
            bounds=lambda m, year: tuple(price_bounds.loc[year, pd.IndexSlice["p_ra", ["lb", "ub"]]].values)
            if price_bounds is not None
            else (0, float("+inf")),
        )

        # Linear interpolation for p_ghg in Non-modeled Years
        @self.Expression(self.YEARS)
        def p_ghg_interp(model, year):
            if year in model.MODELED_YEARS:
                return model.p_ghg[year]
            if year > 2045:
                return model.p_ghg[2045]
            lower_years = [y for y in model.MODELED_YEARS if y < year]
            upper_years = [y for y in model.MODELED_YEARS if y > year]
            if not lower_years or not upper_years:
                raise ValueError("Interpolation years are out of bounds.")
            lower_year = max(lower_years)
            upper_year = min(upper_years)
            fraction = (year - lower_year) / (upper_year - lower_year)
            return model.p_ghg[lower_year] + fraction * (model.p_ghg[upper_year] - model.p_ghg[lower_year])
        
        # Linear interpolation for p_ra in Non-modeled Years
        @self.Expression(self.YEARS)
        def p_ra_interp(model, year):
            if year in model.MODELED_YEARS:
                return model.p_ra[year]
            if year > 2045:
                return model.p_ra[2045]
            lower_years = [y for y in model.MODELED_YEARS if y < year]
            upper_years = [y for y in model.MODELED_YEARS if y > year]
            if not lower_years or not upper_years:
                raise ValueError("Interpolation years are out of bounds.")
            lower_year = max(lower_years)
            upper_year = min(upper_years)
            fraction = (year - lower_year) / (upper_year - lower_year)
            return model.p_ra[lower_year] + fraction * (model.p_ra[upper_year] - model.p_ra[lower_year])

        # Calculate revenue for each resource vintage
        # Calculate RA contribution for each resource vintage
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_ra_revenue_in_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return model.p_ra_interp[year] * self.input_data.loc[(resource, vintage, year), "ra_capacity_mw"] * self.input_data.loc[(resource, vintage, year), "operational_capacity_mw"] * 1000

        # If emitting resource, then calculate emission penalty
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_ghg_cost_in_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return -model.p_ghg_interp[year] * self.input_data.loc[(resource, vintage, year), "emissions_units_per_year"] * self.input_data.loc[(resource, vintage, year), "operational_capacity_mw"]

        # Calculate GHG contribution for each resource vintage
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_ghg_energy_revenue_impact_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return (
                model.p_ghg_interp[year]
                * self.input_data.loc[(resource, vintage, year), "emissions_weighted_generation_tonnes_per_year"] * self.input_data.loc[(resource, vintage, year), "operational_capacity_mw"]
            )

        # Calculate total revenue payment per year
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_payments_in_year(model, resource, vintage, year):
            return (
                model.resource_ra_revenue_in_year[resource, vintage, year]
                + model.resource_ghg_cost_in_year[resource, vintage, year]
                + model.resource_ghg_energy_revenue_impact_year[resource, vintage, year]
            )

        # Calculate net cost per year
        @self.Expression(self.RESOURCE_VINTAGES, self.YEARS)
        def resource_net_costs_in_year(model, resource, vintage, year):
            if year not in self.input_data.loc[pd.IndexSlice[resource, vintage], :].index:
                return 0.0
            return -(
                (self.input_data.loc[(resource, vintage, year), "total_annualized_fixed_costs_dollars_per_yr"] * self.input_data.loc[(resource, vintage, year), "operational_capacity_mw"])
                - (self.input_data.loc[(resource, vintage, year), "net_energy_as_revenue_$_per_year"] * self.input_data.loc[(resource, vintage, year), "operational_capacity_mw"])
            )

        # Calculate net revenue per year
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
            return model.npv_net_revenue[resource, vintage] >= 0

        # NPV first year(s) net revenues  >= 0 if the constraint is enabled
        @self.Constraint(self.RESOURCE_VINTAGES)
        def constrain_first_year_net_revenue(model, resource, vintage):
            M = 1e10
            return (model.first_year_net_revenue[resource, vintage]  >= -M * (1 - model.enable_first_year_revenue_constraint))
        
        # GHG avoided costs constant after 2045
        @self.Constraint(self.YEARS)
        def constant_p_ghg_after_2045(model, year):
            if year > 2045:
                return model.p_ghg_interp[year] == model.p_ghg_interp[year - 1]
            return pyo.Constraint.Skip

        # Capacity avoided costs constant after 2045
        @self.Constraint(self.YEARS)
        def constant_p_ra_after_2045(model, year):
            if year > 2045:
                return model.p_ra_interp[year] == model.p_ra_interp[year - 1]
            return pyo.Constraint.Skip

        
        # ## Objectives
        @self.Objective(sense=pyo.minimize)
        def npv_net_payments(model):
            npv_revenues = sum(model.npv_net_revenue[resource, vintage] for resource, vintage in model.RESOURCE_VINTAGES)
            return npv_revenues

    def update_results_attributes(self):
        self.prices = pd.DataFrame.from_dict(
            {
                col: {year: pyo.value(p[year]) for year in self.YEARS}
                for col, p in {
                    "GHG (${}/tonne)".format(DOLLAR_YEAR): self.p_ghg_interp,
                    "Gen Capacity (${}/kW-yr)".format(DOLLAR_YEAR): self.p_ra_interp,
                }.items()
            }
        )

        self.prices.reset_index(inplace=True)
        self.prices.rename(columns={'index': 'Year'}, inplace=True)

    def solve(self, tee: bool = False):
        if sys.platform=="win32":
            opt = pyo.SolverFactory(
                "cbc",
                solver_io="lp",
                executable="./solvers/cbc.exe",
            )
        else:
            opt = pyo.SolverFactory(
                "cbc",
                solver_io="lp",
            )
        self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        solution = opt.solve(self, tee=tee)

        if solution.solver.termination_condition == pyo.TerminationCondition.optimal:
            logger.info(f"Objective function value (NPV net payments): ${pyo.value(self.npv_net_payments):,.2f}")

            self.update_results_attributes()

        elif solution.solver.termination_condition == pyo.TerminationCondition.infeasible:
            logger.critical("Model infeasible.")

    # Create plots
    def plot_avoided_costs(self, run_folder, first_years_profitable_constraint, first_years_profitable):

        prices_long_df = self.prices.melt(id_vars='Year', var_name='variable', value_name='value')

        fig = px.line(
        prices_long_df,
        x='Year',
        y='value',
        color='variable', 
        color_discrete_sequence=['green', 'red'], 
        labels={'Value': 'Cost ($)', 'Year': 'Year', 'Type': 'Cost Type'}
)

        fig.update_layout(template="plotly_white", height=400, width=800)
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
            .loc[:, ["total_annualized_fixed_costs_dollars_per_yr", "net_energy_as_revenue_$_per_year", "operational_capacity_mw"]]
            .rename(
                columns={
                    "total_annualized_fixed_costs_dollars_per_yr": "Fixed Cost per MW",
                    "net_energy_as_revenue_$_per_year": "Net Energy + AS Revenue per MW",
                    "operational_capacity_mw": "Operational Capacity"
                }
            )
        )

        # Calculate "Net Energy + AS Revenue" and "Fixed Costs"
        resource_inputs["Net Energy + AS Revenue"] = (
        resource_inputs["Net Energy + AS Revenue per MW"] * resource_inputs["Operational Capacity"]
        )

        resource_inputs["Fixed Cost"] = (
        resource_inputs["Fixed Cost per MW"] * resource_inputs["Operational Capacity"]
        )

        # Concatenate the calculated columns with resource_cash_flow
        resource_cash_flow = pd.concat(
            [
                resource_inputs,
                resource_cash_flow.reindex(resource_inputs.index),
            ],
            axis=1,
        ).sort_index()

        return resource_cash_flow
    
    # Calculate cashflow for all resources and all vintages
    def calculate_all_resource_cashflow(self):
        all_resource_cashflow = pd.DataFrame()
        
        for resource, vintage in self.RESOURCE_VINTAGES:
            cashflow = self.calculate_one_resource_cash_flow(resource_vintage=(resource, vintage))

            # Create a DataFrame for the current vintage's cash flow data
            cashflow_df = pd.DataFrame(cashflow, columns=cashflow.columns.values)

            # Add 'Resource' and 'Vintage' columns to the DataFrame and set their values
            cashflow_df["Resource"] = resource
            cashflow_df["Vintage"] = vintage

            # Concatenate the current vintage's DataFrame with the main DataFrame
            all_resource_cashflow = pd.concat([all_resource_cashflow, cashflow_df])
        
        return all_resource_cashflow
    
    # Calculate NPV revenues and costs for all resources
    def calculate_all_resource_npv(self):
        discount_factors = (
            self.input_data.loc[:, "discount_factor"].reset_index(["resource", "vintage"], drop=True).drop_duplicates()
        )

        npv_results = {}
        for resource, vintage in self.RESOURCE_VINTAGES:
            cashflow = self.calculate_one_resource_cash_flow(resource_vintage=(resource, vintage))

            # Calculate individual NPVs
            npv_gen_cap = cashflow['Gen Cap Value'].mul(discount_factors.reindex(cashflow.index), axis=0).sum()
            npv_ghg_cost = cashflow['GHG Cost'].mul(discount_factors.reindex(cashflow.index), axis=0).sum()
            npv_ghg_value = cashflow['GHG Value'].mul(discount_factors.reindex(cashflow.index), axis=0).sum()
            npv_net_energy = cashflow['Net Energy + AS Revenue'].mul(discount_factors.reindex(cashflow.index), axis=0).sum()

            # Total NPV from all components
            npv_total = npv_gen_cap + npv_ghg_cost + npv_ghg_value + npv_net_energy

            # Calculate NPV of fixed costs for cost recovery calculation
            npv_fixed_costs = cashflow['Fixed Cost'].mul(discount_factors.reindex(cashflow.index), axis=0).sum()

            # Calculate the percentage of cost recovery
            if npv_fixed_costs != 0:
                cost_recovery_percentage = (npv_total / npv_fixed_costs)
            else:
                cost_recovery_percentage = float('nan')  # Handle division by zero

            # Store all results
            npv_results[(resource, vintage)] = {
                'Fixed Cost': npv_fixed_costs,
                'Gen Cap Value': npv_gen_cap,
                'GHG Cost': npv_ghg_cost,
                'GHG Value': npv_ghg_value,
                'Net Energy + AS Revenue': npv_net_energy,
                'Total Value': npv_total,
                '% of Cost Recovery': cost_recovery_percentage
            }

        # Convert the results dictionary to DataFrame
        all_resource_npv = pd.DataFrame.from_dict(npv_results, orient='index')
        all_resource_npv.index.names = ['resource', 'vintage']

        return all_resource_npv

    # Plot cashflow for one resource and one vintage
    def plot_resource_cash_flow(self, *, resource_vintage: tuple):
        resource, vintage = resource_vintage
        cash_flow = self.calculate_one_resource_cash_flow(resource_vintage=(resource, vintage))
        columns_to_plot = ["Net Energy + AS Revenue", "GHG Value", "Gen Cap Value", "GHG Cost"]

        fig = px.bar(
            cash_flow,
            x=cash_flow.index,
            y=columns_to_plot,
            title="{} Vintage {} Cash Flow ($)".format(resource, vintage),
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
    def plot_npv_net_revenues(self, *, resource, columns_to_plot=None):
        resource_npv = self.calculate_all_resource_npv().loc[resource]

        if columns_to_plot is None:
            columns_to_plot = resource_npv.columns[1:]

        fig = px.bar(
            resource_npv,
            x=resource_npv.index,
            y=columns_to_plot,
            title="{} NPV by Vintage ($/MW Nameplate)".format(resource),
            barmode="stack",
        )

        fig.add_scatter(
            x=resource_npv.index,
            y=resource_npv["Fixed Cost"],
            mode="markers",
            name='Fixed Cost',
            showlegend=True,
            marker=dict(color="#bcbcbc"),
            row=1,
            col=1,
        )
        
        fig.update_layout(
            template="plotly_white", 
            height=400, 
            width=1000,
            yaxis=dict(
            title="${}/MW".format(DOLLAR_YEAR),
            titlefont_size=16,
            tickfont_size=14,
        ),
        )

        return fig


# # Running Model

def main():
    global base_path
    base_path = pathlib.Path(__file__).parent.parent


    # Create output folder
    outputs_file_path = base_path / "results" / RUN_ID
    outputs_file_path.mkdir(exist_ok=True, parents=True)

    # Read resource data csv
    df = read_resource_data(filepath=base_path / "data" / "processed" / RUN_ID / "resource_data.csv")
    df = add_discount_factors(df=df, discount_rate=DISCOUNT_RATE, dollar_year=DOLLAR_YEAR)

    # Run model
    m = AvoidedCostModel(
        input_data=df,
        first_years_profitable=FIRST_YEARS_PROFITABLE,
        price_bounds=pd.read_csv(base_path / "data" / "processed" / RUN_ID / "price_bounds.csv", index_col=0, header=[0, 1])
    )

    m.solve()

    # Output results
    m.prices.to_csv(outputs_file_path / "avoided_costs.csv")
    m.calculate_all_resource_cashflow().to_csv(outputs_file_path / "cash_flow_by_resource.csv")
    m.calculate_all_resource_npv().to_csv(outputs_file_path / "npv_by_resource.csv")


    # # Plot Results
    m.plot_avoided_costs(run_folder=RUN_ID, first_years_profitable_constraint=FIRST_YEARS_PROFITABLE_CONSTRAINT, first_years_profitable=FIRST_YEARS_PROFITABLE)


if __name__ == "__main__":
    main()
