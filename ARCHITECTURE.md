# First-Principles NBA Predictive Model Architecture

A strategic blueprint for transitioning from a rolling-average feature model to a rigorous, possession-based, opponent-adjusted basketball model.

---

## 1. The Paradigm Shift

Your current `NBA-Prop-Maker` relies heavily on **rolling averages** (L5, L10, L20) and engineered features. While profitable, rolling averages are inherently noisy. They treat a 30-point game against the elite Timberwolves defense the same as a 30-point game against the tanking Wizards. They also treat a 120-possession track meet the same as a 90-possession grind.

To build a truly elite model, you must move to **first principles**. A basketball game is simply a finite sequence of possessions. The goal of a model is to predict how many possessions will occur, and what the expected value (Points Per Possession) of each possession will be, given the specific players on the court.

---

## 2. The Foundation: Adjusted Points Per Possession (PPP)

Everything starts with pace normalization. You cannot predict raw points; you must predict **Pace (Possessions)** and **Efficiency (PPP)** separately, then multiply them together.

### The Math
A possession ends in one of four ways: a made shot, a missed shot that is rebounded by the defense, a turnover, or a final free throw.
*   **Estimated Possessions** = `FGA + 0.44 * FTA - ORB + TOV`
*   **Offensive Rating (PPP * 100)** = `(Points / Possessions) * 100`

**Implementation:**
Instead of tracking a player's "Points per Game", you track their "Points per 100 Possessions" while they are on the floor, and their "Usage Rate" (the percentage of team possessions they finish).

---

## 3. Opponent Quality Adjustment

A player's PPP is meaningless without context. If a player averages 1.20 PPP, but they've only played bottom-5 defenses, their true talent is lower.

### The Methodology (Ridge Regression / Bayesian Approach)
You must build a system that simultaneously solves for every team's (or player's) true offensive and defensive ratings.
*   `Actual_PPP = League_Average_PPP + Offense_Strength + Defense_Strength + Home_Court_Advantage`
*   By running a Ridge Regression (or Bayesian Hierarchical Model) across the entire season's play-by-play data, you isolate the **true offensive talent** of a player from the **true defensive talent** of the opponents they faced.

**Implementation:**
When predicting a prop, you don't just look at what the player did last week. You calculate: `Player_Base_PPP + Opponent_Defensive_Adjustment = Expected_Matchup_PPP`.

---

## 4. Garbage Time & Blowout Discounting

NBA teams change their behavior drastically when a game is out of reach. Star players sit, bench players gun for stats, and defenses play drop coverage to avoid fouls. Including these minutes heavily skews your data.

### The Methodology (Win Probability Filtering)
You must filter out "Garbage Time" from your training data. Garbage time is typically defined as:
*   Score differential ≥ 25 points in the 3rd quarter.
*   Score differential ≥ 15 points in the 4th quarter.
*   Score differential ≥ 10 points in the final 3 minutes.

**Implementation:**
When calculating a player's true PPP and Usage Rate, **exclude all possessions that occurred during garbage time**. A bench player scoring 12 points in the 4th quarter of a 30-point blowout should not impact their projection for a tight game tomorrow.

---

## 5. Player-Level Modeling & Injury Replacement

A team is not a static entity; it is a collection of 5 players on the court. When a star player sits out, their usage (shots and assists) and minutes must be redistributed.

### The Methodology (Usage vs. Efficiency Curves)
*   **Usage Rate (USG%):** The percentage of team plays a player was involved in while on the floor.
*   **The Skill Curve:** As a player's Usage Rate increases (because a star teammate is injured), their Efficiency (PPP) almost always *decreases*.

**Implementation:**
1.  **Baseline:** Calculate every player's baseline USG% and PPP.
2.  **Injury Allocation:** If Player A (30% USG) is out, distribute that 30% among the remaining starters and bench replacements based on their historical tendencies when playing without Player A.
3.  **Efficiency Penalty:** Apply a slight penalty to the PPP of the players absorbing the new usage, as they are now taking harder, forced shots.

---

## 6. Luck Modeling & Regression to the Mean

Sports betting models fail when they overreact to variance. You must mathematically separate **Skill** from **Luck**.

### The Three Pillars of NBA Luck
1.  **Opponent Free Throw Percentage:** Defenses have *zero* control over whether an opponent makes a free throw. If a team's opponents are shooting 70% from the line (league average is ~78%), that team's defensive rating looks artificially good. **Adjustment:** Always substitute actual opponent FT% with the league average FT%.
2.  **Opponent 3-Point Percentage:** Defenses control how many 3-pointers they give up (Attempt Rate), but they have very little control over whether those open shots go in (Make Rate). **Adjustment:** Regress opponent 3PT% heavily toward the league average.
3.  **Shooting Variance (Expected eFG%):** A player shooting 60% on contested mid-range jumpers over a 5-game stretch is lucky, not suddenly elite. **Adjustment:** Use shot-location data (rim, short mid, long mid, 3pt) to calculate an *Expected* Effective Field Goal Percentage, and use that for predictions rather than their actual hot/cold streak.

---

## 7. Roadmap: Integrating this into `NBA-Prop-Maker`

To transition your current setup to this first-principles architecture, follow this sequence:

**Phase 1: The Data Upgrade**
*   Move away from basic box scores. You need **Play-by-Play** data to accurately calculate possessions, garbage time, and exact on/off court splits. (The `nba_api` has play-by-play endpoints).

**Phase 2: The Core Engine**
*   Build the Bayesian/Ridge regression model to calculate Opponent-Adjusted PPP and Pace for all 30 teams.

**Phase 3: The Player Micro-Model**
*   Map out USG% and PPP for every player.
*   Build the "Injury Allocator" function that redistributes minutes and usage when a player is ruled OUT.

**Phase 4: The Luck Adjuster**
*   Clean the training data by neutralizing Opponent FT% and regressing Opponent 3PT%.

This is the exact architecture used by professional syndicates. It removes the noise of scheduling, blowouts, and lucky shooting, leaving you with the purest mathematical projection of a basketball game.
