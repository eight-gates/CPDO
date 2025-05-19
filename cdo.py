import numpy as np
import matplotlib.pyplot as plt

class CPDO:
    """
    Constant Proportion Debt Obligation (CPDO) simulator (based on Dorn 2010).
    Organized for clarity, flexibility, and reusability.
    """
    def __init__(self, spreads, G=5.0, recovery=0.4, r=0.01, coupon=0.0,
                 payments_per_year=4, T=5.0, stop_loss_barrier=0.8, eps=0.05):
        self.spreads = np.array(spreads, dtype=float)
        self.n_paths, self.n_steps = self.spreads.shape
        self.T = T
        self.dt = T / (self.n_steps - 1)
        self.times = np.linspace(0, T, self.n_steps)
        self.G = G
        self.recovery = recovery
        self.r = r
        self.N_tranche = 1.0
        self.coupon = coupon
        self.pay_idx = self._compute_payment_indices(payments_per_year, coupon, T)
        self.stop_loss = stop_loss_barrier
        self.eps = eps
        self.NAV_paths = np.zeros((self.n_paths, self.n_steps))
        self.default_loss_paths = np.zeros((self.n_paths, self.n_steps))
        self.final_losses = np.zeros(self.n_paths)
        self.rating = np.zeros(self.n_paths, dtype=object)
        self._run_simulation()
        self._compute_metrics()

    def _compute_payment_indices(self, payments_per_year, coupon, T):
        if payments_per_year > 0 and coupon > 0:
            period = 1.0 / payments_per_year
            payment_times = np.arange(period, T + 1e-8, period)
            return set(int(round(t / self.dt)) for t in payment_times)
        return set()

    def _run_simulation(self):
        for i in range(self.n_paths):
            self._simulate_path(i)

    def _simulate_path(self, ipath):
        S_path = self.spreads[ipath]
        Sc = S_path[0]
        L = self.G
        RE = L * self.N_tranche
        Account = self.N_tranche
        cum_loss = 0.0

        # Record initial NAV
        self.NAV_paths[ipath, 0] = Account
        self.default_loss_paths[ipath, 0] = cum_loss

        for t in range(1, self.n_steps):
            dt = self.dt
            S_market = S_path[t]

            # (a) Accrue interest on cash
            Account *= (1 + self.r * dt)
            # (b) Accrue spread income (Sc * RE)
            Account += Sc * RE * dt

            # (c) Check for default of reference (hazard = S/(1-R))
            if RE > 0:
                intensity = S_market / (1.0 - self.recovery)
                p_default = intensity * dt
                if np.random.uniform() < p_default:
                    # Default occurs
                    loss = (1.0 - self.recovery) * RE
                    Account -= loss
                    cum_loss += loss
                    RE = 0.0
                    L = 0.0

            # (d) Pay coupon if scheduled
            if t in self.pay_idx:
                Account -= self.coupon * self.N_tranche / len(self.pay_idx)

            # (e) Compute mark-to-market (MtM) of risky position (eq.8 Dorn)
            MtM_factor = 0.0
            if RE > 0:
                # Sum discounted survival factors for all future payment dates
                for pay_idx in self.pay_idx:
                    if pay_idx > t:
                        tau = (pay_idx * dt) - t * dt
                        B = np.exp(-self.r * tau)  # discount factor
                        surv = np.exp(-S_market * tau / (1.0 - self.recovery))
                        MtM_factor += B * surv
            MtM = (Sc - S_market) * RE * MtM_factor

            # (f) Compute NAV = Account + MtM
            NAV = Account + MtM

            # (g) Cash-In rule (Dorn: if NAV > PV(liabilities), invest all in cash)
            #    Here we approximate PV(liabilities) ≈ notional (no interest), so check NAV >= 1.0
            if NAV >= self.N_tranche:
                # Cash-In: unwind and move everything to cash
                RE = 0.0
                L = 0.0
                Account = NAV
                # Fill remaining NAV in path and exit
                self.NAV_paths[ipath, t:] = Account
                self.default_loss_paths[ipath, t:] = cum_loss
                break

            # (h) Cash-Out rule (if shortfall > barrier, immediate default)
            shortfall = self.N_tranche - NAV
            if shortfall > self.stop_loss:
                # Cash-Out: default
                RE = 0.0
                L = 0.0
                Account = NAV
                self.NAV_paths[ipath, t:] = Account
                self.default_loss_paths[ipath, t:] = cum_loss
                break

            # (i) Step 3: Leverage adjustment
            # Compute PV of liabilities (simplified as ≈1) and current NAV
            PVLiab = self.N_tranche
            # Optimal leverage (eq.12 Dorn): L_opt = (PVLiab - NAV)/N * G
            L_opt = max((PVLiab - NAV) / self.N_tranche * self.G, 0.0)
            # Implied leverage (eq.13 Dorn): L_imp = RE / N
            L_imp = RE / self.N_tranche
            # Rebalance if L_imp is within ±ε of L_opt; else keep L_imp (Dorn’s rule):contentReference[oaicite:4]{index=4}
            if abs(L_imp - L_opt) > self.eps * L_opt:
                L_new = L_opt
            else:
                L_new = L_imp
            RE_new = L_new * self.N_tranche
            # Update contracted spread (eq.16 Dorn):contentReference[oaicite:5]{index=5}
            if RE_new > RE and RE_new > 0:
                w = RE / RE_new
            elif RE > 0:
                w = RE_new / RE
            else:
                w = 0.0
            Sc = w * Sc + (1 - w) * S_market
            # Update state for next step
            L = L_new
            RE = RE_new

            # (j) Record NAV and loss
            self.NAV_paths[ipath, t] = NAV
            self.default_loss_paths[ipath, t] = cum_loss

        # After final step, compute final loss = max(0, 1 - final NAV)
        final_nav = self.NAV_paths[ipath, -1]
        self.final_losses[ipath] = max(0.0, self.N_tranche - final_nav)

    def _compute_metrics(self):
        defaulted = self.final_losses > 0
        self.PD = defaulted.mean()
        self.EL = np.mean(self.final_losses)
        self.VaR95 = np.percentile(self.final_losses, 95)
        self.rating_label = self._assign_rating(self.PD, self.EL)
        for i in range(self.n_paths):
            self.rating[i] = self.rating_label

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
            return "CCC"
        # Investigate typical CCC rated PD and EL
        # Relationship between L, c, r, a to PD and EL in simulation terms.
