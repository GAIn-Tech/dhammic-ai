# Solar Panel Monitor & Control App — Design Document

## Overview

A mobile + web application for monitoring and controlling a SolarEdge solar panel system with per-panel visibility and remote shutdown capability.

## Requirements

- **System:** SolarEdge with power optimizers (1-15 panels, single roof)
- **Access:** Homeowner + Installer/SetApp credentials
- **Platforms:** Native mobile (iOS/Android) + Web dashboard
- **Goals:** Real-time control, custom visualization, automation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER DEVICES                           │
├─────────────────────┬───────────────────────────────────────┤
│   React Native App  │         React Web Dashboard           │
│   (iOS / Android)   │         (Browser)                     │
└─────────┬───────────┴─────────────────┬─────────────────────┘
          │                             │
          └──────────┬──────────────────┘
                     ▼
          ┌─────────────────────┐
          │   Backend API       │
          │   (Node.js/Express) │
          │   - Auth            │
          │   - Data caching    │
          │   - Automation      │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│  SolarEdge API  │   │    Database     │
│  (Monitoring +  │   │   (SQLite →     │
│   SetApp)       │   │    Postgres)    │
└─────────────────┘   └─────────────────┘
```

**Why a backend in between:**
- SolarEdge API has rate limits (300/day) — backend caches data
- Stores automation rules and historical data
- Single connection point for both mobile and web
- Keeps API keys secure (not in client code)

## Core Features

### Dashboard (Home Screen)
- Live system overview: total production (kW), daily yield (kWh), current status
- Visual panel grid showing each panel's output with color coding
  - Green = healthy
  - Yellow = underperforming
  - Red = issue or off
- Quick stats: today's production, savings estimate, peak output time

### Panel Detail View
- Current watts, voltage, temperature
- Daily/weekly/monthly production graphs
- Status indicator and alerts (shade detected, optimizer fault, etc.)
- **Shutdown toggle** — switch to turn individual panel on/off

### Control Panel
- Master on/off for entire system
- Per-string controls (if multiple strings)
- Individual panel toggles in list view
- Confirmation dialogs for all shutdown actions (safety)

### Automation Rules
- Threshold rules: "If Panel 3 drops below 50W for 10 minutes, send alert"
- Time-based schedules: "Shut down string B at sunset"
- Simple if/then interface, no coding required

### Settings
- SolarEdge API credentials management
- Notification preferences (push, email)
- Panel layout editor (drag panels to match roof layout)

## Tech Stack

### Mobile App (React Native)
- React Native with Expo
- React Navigation
- React Native Paper (UI components)
- Axios (API client)
- Expo Notifications (push alerts)

### Web Dashboard (React)
- React with Vite
- React Router
- Recharts (graphs)
- Tailwind CSS
- Axios (API client)

### Backend (Node.js)
- Express.js
- SQLite (upgradeable to Postgres)
- node-cron (automation scheduler)
- JWT (authentication)

### Hosting
- Backend: Railway or Render (free tier)
- Web: Vercel (free tier)

## SolarEdge API Integration

### Monitoring API (Read Data)
- `/site/{siteId}/overview` — Current production, daily totals
- `/site/{siteId}/power` — Power output over time
- `/equipment/{siteId}/list` — List of all optimizers/panels
- `/equipment/{siteId}/data` — Per-panel metrics

### SetApp/Installer API (Control)
- Remote shutdown commands via installer credentials
- Interacts through SetApp protocol or local inverter commands

### Control Flow
```
User taps "Shut Off Panel 3"
        ↓
App → Backend → SolarEdge SetApp API
        ↓
Inverter receives command
        ↓
Optimizer for Panel 3 goes to safe mode
        ↓
Backend confirms, updates app UI
```

### Safety Measures
- Confirmation dialog before any shutdown
- Audit log of all control actions (who, when, what)
- Auto-restore option after set time period
- Backend validates all commands before forwarding

## Data Flow & Automation

### Data Collection (backend cron job)
```
Every 5 minutes:
  → Fetch site overview from SolarEdge
  → Fetch per-panel data
  → Store in database
  → Check automation rules
  → Push alerts if triggered
```

### Automation Rule Format
```json
{
  "name": "Low output alert",
  "trigger": {
    "type": "threshold",
    "panel": "Panel-3",
    "metric": "watts",
    "condition": "below",
    "value": 50,
    "duration": "10m"
  },
  "action": {
    "type": "notify",
    "method": "push",
    "message": "Panel 3 producing under 50W for 10 minutes"
  }
}
```

### Supported Triggers
- Threshold (above/below value)
- Time-based (at sunset, at specific time)
- Comparison (panel A vs panel B difference)

### Supported Actions
- Send push notification
- Send email
- Shut down panel/string
- Log event

## Project Structure

```
solar-monitor/
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   │   ├── auth.ts
│   │   │   ├── panels.ts
│   │   │   └── controls.ts
│   │   ├── services/
│   │   │   ├── solaredge.ts
│   │   │   ├── automation.ts
│   │   │   └── notifications.ts
│   │   ├── db/
│   │   │   └── schema.sql
│   │   └── index.ts
│   └── package.json
│
├── web/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── PanelDetail.tsx
│   │   │   ├── Controls.tsx
│   │   │   └── Automation.tsx
│   │   ├── components/
│   │   │   ├── PanelGrid.tsx
│   │   │   ├── PowerChart.tsx
│   │   │   └── ShutdownToggle.tsx
│   │   └── api/
│   └── package.json
│
├── mobile/
│   ├── src/
│   │   ├── screens/
│   │   ├── components/
│   │   └── api/
│   └── package.json
│
└── docs/
    └── plans/
```

## Next Steps

1. Set up monorepo with backend, web, and mobile projects
2. Implement backend API with SolarEdge integration
3. Build web dashboard
4. Build mobile app
5. Deploy and test
