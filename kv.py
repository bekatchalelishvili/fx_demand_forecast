from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.appbar import MDTopAppBar
from kivymd.uix.navigationbar import MDNavigationBar, MDNavigationItem
from kivy.uix.screenmanager import ScreenManager
from kivy.metrics import dp
from kivy.properties import NumericProperty, BooleanProperty, StringProperty
import random

# =========================
# CORE GAME STATE
# =========================
class GameState:
    def __init__(self):
        self.day = 1
        self.money = NumericProperty(1800.0)
        self.net_worth = NumericProperty(1800.0)

        # Vital stats
        self.energy = NumericProperty(100)
        self.max_energy = 100
        self.hunger = NumericProperty(0)
        self.mood = StringProperty("Neutral")

        # Career
        self.job_level = 0
        self.experience = 0
        self.main_skill = NumericProperty(1)

        # Lifestyle
        self.lifestyle = StringProperty("Normal")
        self.daily_expenses_base = 60

        # Assets
        self.assets = {
            "bicycle": False, "motorcycle": False, "sedan": False, "luxury_car": False,
            "small_apartment": False, "house": False,
            "smartphone_upgrade": False, "laptop_pro": False,
        }
        self.asset_values = {
            "bicycle": 300, "motorcycle": 2500, "sedan": 18000, "luxury_car": 65000,
            "small_apartment": 85000, "house": 280000,
            "smartphone_upgrade": 1200, "laptop_pro": 2800,
        }

        # Businesses
        self.businesses = {
            "food_truck": {"owned": False, "level": 0, "daily_profit_range": (60, 180)},
            "coffee_shop": {"owned": False, "level": 0, "daily_profit_range": (120, 320)},
            "online_store": {"owned": False, "level": 0, "daily_profit_range": (80, 400)},
            "tech_startup": {"owned": False, "level": 0, "daily_profit_range": (0, 1200)},
        }

        # Investments
        self.investments = {
            "index_fund": {"amount": 0, "daily_return": 0.0008},
            "crypto": {"amount": 0, "daily_return": 0.0035},
            "real_estate_fund": {"amount": 0, "daily_return": 0.0012},
        }

        # Education & perks
        self.education = {
            "coding_bootcamp": False,
            "gym_membership": False,
            "english_advanced": False,
        }

        self.achievements = set()

    def next_day(self):
        self.day += 1

        multiplier = {"Cheap": 0.6, "Normal": 1.0, "Luxury": 1.8}[self.lifestyle]
        expenses = self.daily_expenses_base * multiplier
        if self.assets["small_apartment"] or self.assets["house"]:
            expenses += 80
        self.money -= expenses
        self.hunger += random.randint(15, 35)

        for data in self.investments.values():
            change = data["amount"] * data["daily_return"] * random.uniform(0.6, 1.4)
            self.money += change

        for b in self.businesses.values():
            if b["owned"]:
                profit = random.randint(*b["daily_profit_range"])
                if random.random() < 0.08:
                    profit = max(0, profit // 3)
                self.money += profit * (1 + b["level"] * 0.25)

        recovery = 45 if self.education["gym_membership"] else 30
        self.energy = min(self.max_energy, self.energy + recovery)

        if self.hunger > 70:
            self.mood = "Stressed"
            self.energy = max(10, self.energy - 12)
        elif self.hunger < 25 and self.energy > 70:
            self.mood = "Happy"
        else:
            self.mood = "Neutral"

        self.update_net_worth()
        self.trigger_random_event()
        self.check_milestones()

    def update_net_worth(self):
        total = self.money
        for k, v in self.assets.items():
            if v:
                total += self.asset_values.get(k, 0)
        for inv in self.investments.values():
            total += inv["amount"]
        self.net_worth = total

    def trigger_random_event(self):
        roll = random.random()
        if roll < 0.08:
            loss = random.randint(150, 600)
            self.money -= loss
            return f"Unexpected expense! -${loss}"
        elif roll < 0.14:
            gain = random.randint(200, 800)
            self.money += gain
            return f"Found side gig! +${gain}"
        elif roll < 0.18 and any(b["owned"] for b in self.businesses.values()):
            msg = random.choice(["Customer boom!", "Bad review wave."])
            if "boom" in msg:
                self.money += random.randint(300, 1200)
            else:
                self.money -= random.randint(200, 700)
            return msg
        return None

    def check_milestones(self):
        thresholds = [
            (5000, "First $5k"), (25000, "Middle Class"),
            (100000, "Six Figures"), (500000, "Half Million"),
            (1000000, "Millionaire"),
        ]
        for val, title in thresholds:
            if self.net_worth >= val and title not in self.achievements:
                self.achievements.add(title)

    def financial_tier(self):
        if self.money < -2000: return "Bankrupt Risk"
        if self.money < 0:     return "In Debt"
        if self.net_worth < 10000:  return "Struggling"
        if self.net_worth < 60000:  return "Working Class"
        if self.net_worth < 250000: return "Middle Class"
        if self.net_worth < 1000000:return "Wealthy"
        return "Elite"

# =========================
# BASE SCREEN
# =========================
class GameScreen(MDScreen):
    def __init__(self, game_state, **kwargs):
        super().__init__(**kwargs)
        self.game = game_state

    def refresh_all(self):
        pass

# =========================
# HOME SCREEN
# =========================
class HomeScreen(GameScreen):
    def __init__(self, game_state, **kwargs):
        super().__init__(game_state, name="home", **kwargs)

        root = MDBoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))

        card = MDCard(
            orientation="vertical",
            padding=dp(16),
            spacing=dp(8),
            radius=[dp(16)],
            elevation=2,
        )
        self.lbl_main = MDLabel(
            text=self.get_main_stats(),
            halign="left",
            markup=True,
            font_style="BodyLarge",
        )
        card.add_widget(self.lbl_main)

        quick = MDGridLayout(cols=2, spacing=dp(12), adaptive_height=True)

        btn_eat = MDButton(style="tonal")
        btn_eat.add_widget(MDButtonText(text="Eat (-$15, -20 hunger)"))
        btn_eat.bind(on_release=self.eat)
        quick.add_widget(btn_eat)

        btn_rest = MDButton(style="tonal")
        btn_rest.add_widget(MDButtonText(text="Rest (+35 energy)"))
        btn_rest.bind(on_release=self.rest)
        quick.add_widget(btn_rest)

        btn_next = MDButton(
            style="filled",
            md_bg_color=(0.85, 0.25, 0.2, 1),
            pos_hint={"center_x": 0.5},
            size_hint_x=0.9,
        )
        btn_next.add_widget(MDButtonText(text="Advance to Next Day"))
        btn_next.bind(on_release=self.advance_day)

        root.add_widget(card)
        root.add_widget(quick)
        root.add_widget(btn_next)
        self.add_widget(root)

    def get_main_stats(self):
        event = self.game.trigger_random_event() or ""
        mood_color = {
            "Happy": "[color=#66ff99]",
            "Neutral": "[color=#eeeeee]",
            "Stressed": "[color=#ff9966]",
        }.get(self.game.mood, "")

        lines = [
            f"[b]Day[/b] {self.game.day:>4}    [b]Mood[/b] {mood_color}{self.game.mood}[/color]",
            f"[b]Money[/b] ${self.game.money:,.0f}",
            f"[b]Net Worth[/b] ${self.game.net_worth:,.0f}    [size=12sp]{self.game.financial_tier()}[/size]",
            f"[b]Energy[/b] {int(self.game.energy)}/{self.game.max_energy}",
            f"[b]Hunger[/b] {int(self.game.hunger)}",
            f"[b]Lifestyle[/b] {self.game.lifestyle}",
        ]
        if self.game.achievements:
            lines.append(f"[i]Achievements:[/i] {', '.join(list(self.game.achievements)[-2:])}")
        if event:
            lines.append(f"\n[color=#ffcc00]{event}[/color]")

        return "\n".join(lines)

    def advance_day(self, *args):
        self.game.next_day()
        self.parent.parent.refresh_all()  # reach Dashboard

    def eat(self, *args):
        if self.game.money >= 15:
            self.game.money -= 15
            self.game.hunger = max(0, self.game.hunger - 20)
            self.game.energy = min(self.game.max_energy, self.game.energy + 8)
            self.parent.parent.refresh_all()

    def rest(self, *args):
        if self.game.energy < self.game.max_energy - 10:
            self.game.energy = min(self.game.max_energy, self.game.energy + 35)
            self.game.hunger += 10
            self.parent.parent.refresh_all()

    def refresh_all(self):
        self.lbl_main.text = self.get_main_stats()

# =========================
# CAREER SCREEN
# =========================
class CareerScreen(GameScreen):
    JOBS = [
        {"name": "Intern", "base": 45, "req": 0, "energy": 50},
        {"name": "Junior Dev", "base": 95, "req": 1, "energy": 55},
        {"name": "Mid Developer", "base": 185, "req": 2, "energy": 60},
        {"name": "Senior Developer", "base": 340, "req": 3, "energy": 65},
        {"name": "Lead / Architect", "base": 580, "req": 4, "energy": 70},
        {"name": "Engineering Mgr", "base": 950, "req": 5, "energy": 75},
    ]

    def __init__(self, game_state, **kwargs):
        super().__init__(game_state, name="career", **kwargs)
        layout = MDBoxLayout(orientation="vertical", padding=dp(16), spacing=dp(10))

        self.lbl_status = MDLabel(text=self.get_status(), markup=True)
        layout.add_widget(self.lbl_status)

        grid = MDGridLayout(cols=1, spacing=dp(8), adaptive_height=True)
        self.work_buttons = []

        for job in self.JOBS:
            btn = MDButton(style="tonal")
            btn.add_widget(MDButtonText(text=f"{job['name']}  (${job['base'] * job['energy'] // 50:,.0f} avg)"))
            btn.disabled = self.game.job_level < job["req"]
            btn.job = job
            btn.bind(on_release=self.work_day)
            grid.add_widget(btn)
            self.work_buttons.append(btn)

        btn_promote = MDButton(style="filled")
        btn_promote.add_widget(MDButtonText(text="Request Promotion ($2500)"))
        btn_promote.bind(on_release=self.try_promote)

        btn_overtime = MDButton(style="outlined")
        btn_overtime.add_widget(MDButtonText(text="Overtime (+50% pay, -25 energy)"))
        btn_overtime.bind(on_release=self.overtime)

        layout.add_widget(grid)
        layout.add_widget(btn_promote)
        layout.add_widget(btn_overtime)
        self.add_widget(layout)

    def get_status(self):
        if self.game.job_level >= len(self.JOBS):
            return "[b]You reached the top![/b]"
        curr = self.JOBS[self.game.job_level]
        return f"[b]Role:[/b] {curr['name']}\n[b]Skill:[/b] {self.game.main_skill}\n[b]XP:[/b] {self.game.experience}/1000"

    def work_day(self, btn):
        job = btn.job
        if self.game.job_level < job["req"] or self.game.energy < job["energy"]:
            return
        pay = job["base"] * self.game.main_skill
        if self.game.assets["laptop_pro"]:
            pay *= 1.15
        if self.game.assets["luxury_car"]:
            pay *= 1.08
        self.game.money += pay
        self.game.experience += 80 + random.randint(0, 40)
        self.game.energy -= job["energy"]
        self.game.hunger += random.randint(12, 28)
        self.parent.parent.refresh_all()

    def try_promote(self, *args):
        cost = 2500
        req_skill = self.game.job_level + 2
        if (self.game.money >= cost and self.game.experience >= 800 and
                self.game.main_skill >= req_skill and self.game.job_level < len(self.JOBS) - 1):
            self.game.money -= cost
            self.game.job_level += 1
            self.game.main_skill += 1
            self.game.experience -= 600
            self.parent.parent.refresh_all()

    def overtime(self, *args):
        if self.game.energy < 40 or self.game.job_level >= len(self.JOBS):
            return
        pay = self.JOBS[self.game.job_level]["base"] * self.game.main_skill * 1.5
        self.game.money += pay
        self.game.energy -= 25
        self.game.hunger += 18
        self.parent.parent.refresh_all()

    def refresh_all(self):
        self.lbl_status.text = self.get_status()
        for btn in self.work_buttons:
            job = btn.job
            btn.disabled = self.game.job_level < job["req"] or self.game.energy < job["energy"]

# =========================
# ASSETS & LIFESTYLE SCREEN
# =========================
class AssetsScreen(GameScreen):
    def __init__(self, game_state, **kwargs):
        super().__init__(game_state, name="assets", **kwargs)
        layout = MDBoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))

        self.lbl_summary = MDLabel(text=self.get_summary(), markup=True)
        layout.add_widget(self.lbl_summary)

        grid = MDGridLayout(cols=1, spacing=dp(8), adaptive_height=True)

        items = [
            ("Bicycle", 800, "bicycle"),
            ("Motorcycle", 4800, "motorcycle"),
            ("Sedan", 22000, "sedan"),
            ("Luxury Car", 72000, "luxury_car"),
            ("Small Apartment", 95000, "small_apartment"),
            ("House", 320000, "house"),
            ("Pro Laptop", 3200, "laptop_pro"),
            ("Flagship Phone", 1400, "smartphone_upgrade"),
        ]

        for name, price, key in items:
            owned = self.game.assets[key]
            style = "outlined" if owned else "filled"
            text = f"{name} (owned)" if owned else f"Buy {name} ${price:,}"
            btn = MDButton(style=style, disabled=owned or self.game.money < price)
            btn.add_widget(MDButtonText(text=text))
            btn.key = key
            btn.price = price
            btn.bind(on_release=self.buy_asset)
            grid.add_widget(btn)

        lifestyle_grid = MDGridLayout(cols=3, spacing=dp(8), adaptive_height=True)
        for lvl in ["Cheap", "Normal", "Luxury"]:
            style = "tonal" if lvl == self.game.lifestyle else "outlined"
            btn = MDButton(style=style)
            btn.add_widget(MDButtonText(text=lvl))
            btn.lvl = lvl
            btn.bind(on_release=self.change_lifestyle)
            lifestyle_grid.add_widget(btn)

        layout.add_widget(grid)
        layout.add_widget(MDLabel(text="Lifestyle Level:", font_style="TitleMedium"))
        layout.add_widget(lifestyle_grid)
        self.add_widget(layout)

    def get_summary(self):
        lines = ["[b]Owned Assets[/b]"]
        for k, owned in self.game.assets.items():
            if owned:
                lines.append(f"• {k.replace('_', ' ').title()}")
        lines.append(f"\n[b]Daily expenses:[/b] ~${self.game.daily_expenses_base * {'Cheap':0.6,'Normal':1.0,'Luxury':1.8}[self.game.lifestyle]:.0f}")
        return "\n".join(lines)

    def buy_asset(self, btn):
        if self.game.money >= btn.price and not self.game.assets[btn.key]:
            self.game.money -= btn.price
            self.game.assets[btn.key] = True
            self.game.update_net_worth()
            self.parent.parent.refresh_all()

    def change_lifestyle(self, btn):
        self.game.lifestyle = btn.lvl
        self.parent.parent.refresh_all()

    def refresh_all(self):
        self.lbl_summary.text = self.get_summary()

# =========================
# BUSINESS & INVESTMENTS SCREEN
# =========================
class BusinessScreen(GameScreen):
    def __init__(self, game_state, **kwargs):
        super().__init__(game_state, name="business", **kwargs)
        layout = MDBoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))

        self.lbl_passive = MDLabel(text=self.get_passive_income(), markup=True)
        layout.add_widget(self.lbl_passive)

        grid = MDGridLayout(cols=1, spacing=dp(8), adaptive_height=True)

        for name, data in self.game.businesses.items():
            owned = data["owned"]
            lvl = data["level"]
            cost = 5000 * (2 ** lvl) if owned else [5000, 8000, 15000, 35000, 80000][list(self.game.businesses.keys()).index(name)]
            text = f"Upgrade {name.title()} (Lv {lvl}) - ${cost:,}" if owned else f"Start {name.title()} - ${cost:,}"
            style = "filled" if owned else "tonal"
            btn = MDButton(style=style, disabled=(not owned and self.game.money < cost))
            btn.add_widget(MDButtonText(text=text))
            btn.name = name
            btn.cost = cost
            btn.bind(on_release=self.handle_business)
            grid.add_widget(btn)

        for inv_name in ["index_fund", "crypto", "real_estate_fund"]:
            amt = self.game.investments[inv_name]["amount"]
            btn = MDButton(style="tonal")
            btn.add_widget(MDButtonText(text=f"Invest +$5000 in {inv_name.replace('_',' ').title()} (${amt:,})"))
            btn.inv = inv_name
            btn.bind(on_release=self.invest_more)
            grid.add_widget(btn)

        layout.add_widget(grid)
        self.add_widget(layout)

    def get_passive_income(self):
        lines = ["[b]Passive Income (monthly est.)[/b]"]
        total = 0
        for inv, data in self.game.investments.items():
            est = data["amount"] * data["daily_return"] * 30
            total += est
            lines.append(f"• {inv.replace('_',' ').title()}: ~${est:,.0f}")
        bus_profit = sum(random.randint(*d["daily_profit_range"]) for d in self.game.businesses.values() if d["owned"]) * 30
        lines.append(f"• Businesses: ~${bus_profit:,.0f}")
        lines.append(f"\n[b]Total monthly passive ≈ ${total + bus_profit:,.0f}[/b]")
        return "\n".join(lines)

    def handle_business(self, btn):
        data = self.game.businesses[btn.name]
        if not data["owned"]:
            if self.game.money >= btn.cost:
                self.game.money -= btn.cost
                data["owned"] = True
                data["level"] = 1
        else:
            if self.game.money >= btn.cost:
                self.game.money -= btn.cost
                data["level"] += 1
        self.parent.parent.refresh_all()

    def invest_more(self, btn):
        if self.game.money >= 5000:
            self.game.money -= 5000
            self.game.investments[btn.inv]["amount"] += 5000
            self.game.update_net_worth()
            self.parent.parent.refresh_all()

    def refresh_all(self):
        self.lbl_passive.text = self.get_passive_income()

# =========================
# MAIN DASHBOARD
# =========================
class LifeSimDashboard(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.game = GameState()

        self.top = MDTopAppBar(
            title=f"Life Sim – Day {self.game.day}",
            elevation=4,
        )
        self.add_widget(self.top)

        self.sm = ScreenManager()
        self.sm.add_widget(HomeScreen(self.game))
        self.sm.add_widget(CareerScreen(self.game))
        self.sm.add_widget(AssetsScreen(self.game))
        self.sm.add_widget(BusinessScreen(self.game))
        self.add_widget(self.sm)

        bottom = MDNavigationBar()
        for name, icon, screen_name in [
            ("Home", "home", "home"),
            ("Career", "briefcase", "career"),
            ("Assets", "car-estate", "assets"),
            ("Business", "finance", "business"),
        ]:
            item = MDNavigationItem()
            item.add_widget(MDNavigationItemIcon(icon=icon))
            item.add_widget(MDNavigationItemLabel(text=name))
            item.screen_name = screen_name
            item.bind(on_release=lambda x: setattr(self.sm, "current", x.screen_name))
            bottom.add_widget(item)

        self.add_widget(bottom)

    def refresh_all(self):
        self.top.title = f"Life Sim – Day {self.game.day}"
        for screen in self.sm.screens:
            if hasattr(screen, "refresh_all"):
                screen.refresh_all()

# =========================
# APP
# =========================
class LifeSimApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_color = (0.25, 0.32, 0.71, 1)   # some blue-gray-ish color
        self.theme_cls.accent_color = (1.0, 0.75, 0.0, 1)      # amber
        return LifeSimDashboard()

if __name__ == "__main__":
    LifeSimApp().run()