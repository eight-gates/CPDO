from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

class CPDO:
    """
    A class to simulate the full lifecycle of a Constant Proportion Debt Obligation (CPDO).

    This class takes in Monte Carlo simulated CDS spread paths for a 5-name basket and simulates 
    the CPDO strategy, including dynamic leverage adjustment, premium accrual, default events, 
    and cash-in/cash-out triggers. It produces the NAV path and loss metrics for each scenario.
    """

    def __init__(self, initial_nav=100.0, coupon_rate=0.05, cushion_rate=0.01,
                 recovery_rate=0.4, max_leverage=15.0, cash_in_frac=1.0, cash_out_frac=0.1,
                 default_model='hazard', time_horizon_years=10, coupon_freq=2):
        """
        Initialize the CPDO with given parameters.

        Parameters:
        - initial_nav: Initial Net Asset Value (principal) of the CPDO note (e.g., 100).
        - coupon_rate: Annual coupon rate promised to investors (e.g., 0.05 for 5% per annum).
        - cushion_rate: Additional annual yield target to build reserves (e.g., 0.01 for 1% extra yield).
        - recovery_rate: Recovery rate on defaulted notional (e.g., 0.4 means 40% recovery, 60% loss).
        - max_leverage: Maximum allowed leverage (notional exposure / NAV), e.g., 15.
        - cash_in_frac: NAV fraction (of initial) to trigger cash-in (e.g., 1.0 = 100% of initial principal).
        - cash_out_frac: NAV fraction (of initial) to trigger cash-out (e.g., 0.1 = 10% of initial principal).
        - default_model: 'hazard' for hazard-based default simulation, or 'barrier' for threshold-based.
        - time_horizon_years: Total time horizon for simulation (e.g., 10 years).
        - coupon_freq: Number of coupon payments per year (e.g., 2 for semiannual coupons).
        """
        # Store parameters
        self.initial_nav = initial_nav
        self.coupon_rate = coupon_rate
        self.cushion_rate = cushion_rate  # is it required to considered at this stage? full sim as a bank
        self.recovery_rate = recovery_rate # Static recovery rate
        self.max_leverage = max_leverage # Static max leverage
        self.cash_in_frac = cash_in_frac
        self.cash_out_frac = cash_out_frac
        self.default_model = default_model
        self.time_horizon_years = time_horizon_years
        self.coupon_freq = coupon_freq

        # Prepare coupon schedule (times in years when coupons are paid)
        if coupon_freq > 0:
            # E.g., for semiannual, coupon_freq=2 -> payments at 0.5, 1.0, 1.5, ... years up to horizon
            # what is this 1e-8 number ?
            self.coupon_schedule = np.arange(1.0 / coupon_freq, time_horizon_years + 1e-8, 1.0 / coupon_freq)
        else:
            self.coupon_schedule = np.array([])

        # Target annual spread income needed (coupon + cushion, on initial principal)
        # This remains constant in our model (could be adjusted over time if desired).
        self.target_yield = coupon_rate + cushion_rate

        # Placeholders for results (filled after simulation)
        self.nav_paths = None  # NAV trajectory for each scenario
        self.default_loss_paths = None  # Cumulative default losses for each scenario over time
        self.final_losses = None  # Final principal loss per scenario
        self.PD = None  # Probability of default (note fails to repay fully)
        self.EL = None  # Expected loss (as fraction of initial principal)
        self.VaR95 = None  # 95% Value-at-Risk of loss
        self.VaR99 = None  # 99% Value-at-Risk of loss
        self.rating = None  # Implied rating based on PD/EL

    def simulate(self, spread_paths):
        """
        Run the CPDO simulation on a set of CDS spread paths.

        Parameters:
        - spread_paths: NumPy array of shape (n_scenarios, T, n_names) containing simulated daily spreads
                        for each name in the basket. Spreads should be in **decimal** form (e.g., 0.01 for 100 bps).

        Returns:
        A dictionary with keys:
        - "NAV_paths": Array (n_scenarios x T+1) of NAV over time for each scenario.
        - "default_loss_paths": Array (n_scenarios x T+1) of cumulative default loss over time for each scenario.
        - "final_losses": Array (n_scenarios,) of final tranche loss (principal shortfall) for each scenario.
        - "PD": Probability of default (fraction of scenarios with final loss > 0).
        - "EL": Expected loss (average final loss as a fraction of initial principal).
        - "VaR95": 95% Value-at-Risk of the loss distribution.
        - "VaR99": 99% Value-at-Risk of the loss distribution.
        - "rating": Implied credit rating based on the PD/EL (relative to agency standards).
        """
        spread_paths = np.array(spread_paths) # TODO: is a copy needed?
        n_scenarios, T, n_names = spread_paths.shape
        # Time step (in years) assuming spreads cover the full horizon uniformly
        dt = self.time_horizon_years / T # TODO: How about T/255? 5/1207 => time step divided in equal parts

        # Initialize result arrays
        nav_paths = np.zeros((n_scenarios, T + 1))
        default_loss_paths = np.zeros((n_scenarios, T + 1))

        # Loop through each scenario path
        for i in range(n_scenarios):
            nav = self.initial_nav  # current NAV for this scenario
            alive = np.ones(n_names, dtype=bool)  # which names have not defaulted
            cash_in = False
            cash_out = False
            nav_paths[i, 0] = nav
            default_loss_paths[i, 0] = 0.0

            # Simulate day-by-day
            for t in range(T):
                # If already cashed-in or cashed-out, keep NAV constant for remaining period
                # TODO: Would probably want to grow NAV at risk free to counter balance losses?
                if cash_in or cash_out:
                    nav_paths[i, t + 1] = nav
                    default_loss_paths[i, t + 1] = default_loss_paths[i, t]
                    continue

                # Get current spreads for this day (for all names)
                current_spreads = spread_paths[i, t]  # shape (n_names,)
                # Calculate average spread of alive names (for leverage calculation)
                alive_spreads = current_spreads[alive]
                if alive_spreads.size == 0:
                    # All names defaulted – no further premium can be earned; NAV remains as is
                    nav_paths[i, t + 1:] = nav
                    default_loss_paths[i, t + 1:] = default_loss_paths[i, t]
                    break
                avg_spread = np.mean(alive_spreads)

                # Determine required leverage to achieve target spread income
                # Target annual cash (spread income) needed = target_yield * initial_nav
                target_annual_cash = self.target_yield * self.initial_nav
                # Required leverage (before cap) = target_annual_cash / (avg_spread * current NAV)
                required_leverage = target_annual_cash / (avg_spread * nav) if avg_spread > 0 else self.max_leverage
                leverage = min(required_leverage, self.max_leverage)  # apply leverage cap
                # Total CDS notional to sell (exposure)
                total_exposure = leverage * nav
                # Distribute exposure equally among alive names (for simplicity)
                exposure_per_name = total_exposure / alive_spreads.size

                # --- Default simulation for this step ---
                defaults_this_step = np.zeros(n_names, dtype=bool)
                if self.default_model == 'hazard':
                    # Hazard-based default: random draw based on default intensity = spread/(1-Recovery)
                    for j in range(n_names):
                        if alive[j]:
                            lam = current_spreads[j] / max(1e-6, (1 - self.recovery_rate))
                            # TODO: Needs the different recovery rates
                            p_default = 1 - np.exp(-lam * dt)  # default probability in this interval
                            if np.random.rand() < p_default: # TODO: WHATS THISSS ???
                                defaults_this_step[j] = True
                elif self.default_model == 'barrier':
                    # Barrier-based default: if spread exceeds a high threshold, default is triggered
                    # (Example threshold: 1000 bps)
                    # TODO: thresholding technique from today's lecture than using fixed threshold, EVT, excess GPD
                    threshold = 0.10  # 10% spread as default barrier
                    for j in range(n_names):
                        if alive[j] and current_spreads[j] >= threshold:
                            defaults_this_step[j] = True

                # Apply losses from defaults
                if np.any(defaults_this_step):
                    loss_this_step = 0.0
                    for j in range(n_names):
                        if defaults_this_step[j] and alive[j]:
                            # Loss = (1 - recovery_rate) * exposure on that name
                            loss_this_step += (1 - self.recovery_rate) * exposure_per_name
                            alive[j] = False  # mark name as defaulted (no longer alive)
                    nav -= loss_this_step  # reduce NAV by default losses
                    default_loss_paths[i, t + 1] = default_loss_paths[i, t] + loss_this_step
                else:
                    # No default this step
                    default_loss_paths[i, t + 1] = default_loss_paths[i, t]

                # If NAV <= 0 after defaults, trigger cash-out (note defaults completely)
                if nav <= 0:
                    nav = 0.0
                    cash_out = True
                    nav_paths[i, t + 1:] = nav
                    default_loss_paths[i, t + 1:] = default_loss_paths[i, t + 1]
                    break

                # --- Premium accrual for this step ---
                # Earn spread premium on each surviving exposure over the interval
                # TODO: WHAT IS PREMIUM INCOME?
                premium_income = np.sum(exposure_per_name * alive_spreads * dt)
                nav += premium_income  # add premium to NAV

                # --- Mark-to-market adjustment for spread moves (if not the last day) ---
                if t < T - 1:
                    next_spreads = spread_paths[i, t + 1]
                    # Assume a fixed duration for sensitivity of exposure to spread changes (e.g., 5 years)
                    duration = 5.0
                    mtm_change = 0.0
                    for j in range(n_names):
                        if alive[j]:
                            ds = next_spreads[j] - current_spreads[j]
                            # If spreads widen (ds > 0), CPDO loses value (negative change to NAV); if tighten, NAV gains.
                            #TODO: NEEDS TO CHANGE, ZERO COUPON EARNINGS MISSING
                            mtm_change += -ds * duration * exposure_per_name
                    nav += mtm_change  # apply mark-to-market change to NAV

                # --- Check Cash-in trigger ---
                current_time = (t + 1) * dt  # years elapsed up to this point
                # Calculate remaining coupon payments and principal due after this time
                remaining_coupons = np.sum(self.coupon_schedule > current_time)
                remaining_coupon_payments = remaining_coupons * (
                            self.coupon_rate * self.initial_nav / (self.coupon_freq if self.coupon_freq else 1))

                # TODO : WRONG TBP calculation
                total_obligations = self.initial_nav + remaining_coupon_payments  # principal + remaining coupons

                # TODO: Should be made on shortfall
                if nav >= total_obligations:
                    # Sufficient assets to cover all future payments -> cash-in
                    cash_in = True
                    # TODO: WHY?
                    nav = total_obligations  # lock NAV to exactly meet obligations (excess considered set aside)
                    nav_paths[i, t + 1:] = nav
                    default_loss_paths[i, t + 1:] = default_loss_paths[i, t + 1]
                    continue  # proceed to fill out rest of timeline as constant NAV

                # --- Check Cash-out trigger ---
                if nav < self.cash_out_frac * self.initial_nav:
                    # NAV fell below critical threshold -> likely cannot recover
                    cash_out = True
                    nav = max(nav, 0.0)  # ensure non-negative
                    nav_paths[i, t + 1:] = nav
                    default_loss_paths[i, t + 1:] = default_loss_paths[i, t + 1]
                    continue  # end risk-taking for this scenario

                # --- Coupon payment at scheduled dates ---
                if self.coupon_freq > 0:
                    # If a coupon payment is due at this time step, pay it out from NAV
                    if np.any(np.isclose(current_time, self.coupon_schedule, atol=1e-8)):
                        coupon_payment = (self.coupon_rate * self.initial_nav) / self.coupon_freq
                        nav -= coupon_payment
                        # If paying coupon causes NAV to drop below 0, trigger cash-out (default on coupon)
                        if nav < 0:
                            nav = 0.0
                            cash_out = True
                            nav_paths[i, t + 1:] = nav
                            default_loss_paths[i, t + 1:] = default_loss_paths[i, t + 1]
                            continue

                # Record NAV for this step
                nav_paths[i, t + 1] = nav
        # End of scenario loop

        # Compute final outcomes
        final_losses = np.maximum(self.initial_nav - nav_paths[:, -1], 0.0)  # principal shortfall in each scenario
        PD = np.mean(final_losses > 1e-6)  # Probability of default (note fails to repay fully)
        EL = np.mean(final_losses) / self.initial_nav  # Expected loss (fraction of initial)
        VaR95 = np.percentile(final_losses, 95)  # 95th percentile loss
        VaR99 = np.percentile(final_losses, 99)  # 99th percentile loss
        rating = self._assign_rating(PD, EL)  # Map PD/EL to an indicative rating

        # Store results in the object
        self.nav_paths = nav_paths
        self.default_loss_paths = default_loss_paths
        self.final_losses = final_losses
        self.PD = PD
        self.EL = EL
        self.VaR95 = VaR95
        self.VaR99 = VaR99
        self.rating = rating

        self.report_cpdo_mc_error()

        return {
            "NAV_paths": nav_paths,
            "default_loss_paths": default_loss_paths,
            "final_losses": final_losses,
            "PD": PD,
            "EL": EL,
            "VaR95": VaR95,
            "VaR99": VaR99,
            "rating": rating
        }


    def report_cpdo_mc_error(self):
        """Compute and display SE and 95% CI for PD, EL, VaR95, & VaR99."""
        # Prepare data
        defaults = (self.final_losses > 1e-6).astype(int)
        metrics_data = {
            'PD': defaults,
            'EL': self.final_losses,
            'VaR95': self.final_losses,
            'VaR99': self.final_losses
        }
        funcs = {
            'PD': lambda x: np.mean(x),
            'EL': lambda x: np.mean(x),
            'VaR95': lambda x: np.percentile(x, 95),
            'VaR99': lambda x: np.percentile(x, 99)
        }
        se = {}
        ci = {}
        for name, arr in metrics_data.items():
            res = bootstrap((arr,), funcs[name],
                            confidence_level=0.95,
                            n_resamples=2000,
                            method='percentile')
            se[name] = arr.std(ddof=1) / np.sqrt(len(arr))
            low, high = res.confidence_interval
            ci[name] = (low, high)

        # Print results
        print("CPDO Monte Carlo Errors: ")
        for name in funcs.keys():
            low, high = ci[name]
        print(f"  {name:<5s} SE={se[name]:.4f}, 95% CI=({low:.4f}, {high:.4f})")

        # Bar chart of SEs
        names = list(funcs.keys())
        ses = [se[n] for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, ses)
        ax.set_title('CPDO Monte Carlo Standard Errors')
        ax.set_ylabel('Standard Error')
        plt.tight_layout();
        plt.show()

    def _assign_rating(self, PD, EL):
        """
        Internal helper to assign an approximate credit rating based on PD and EL.
        Uses a simplified mapping inspired by rating agency standards.
        """
        # Note: These thresholds are illustrative and not official rating criteria.
        if PD < 0.001 and EL < 0.0005:
            return "AAA"
        elif PD < 0.005 and EL < 0.001:
            return "AA"
        elif PD < 0.01 and EL < 0.002:
            return "A"
        elif PD < 0.02 and EL < 0.005:
            return "BBB"
        elif PD < 0.05 and EL < 0.01:
            return "BB"
        elif PD < 0.10 and EL < 0.02:
            return "B"
        else:
            return "CCC or below"
