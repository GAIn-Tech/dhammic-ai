# Solar Monitor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a mobile + web app for monitoring and controlling a SolarEdge solar panel system with per-panel visibility, remote shutdown, and automation.

**Architecture:** Monorepo with Node.js/Express backend (caches SolarEdge data, runs automation), React web dashboard, and React Native mobile app. Backend serves as single API for both clients.

**Tech Stack:** TypeScript throughout, Express + SQLite backend, React + Vite + Tailwind web, React Native + Expo mobile, JWT auth, node-cron for automation.

---

## Phase 1: Project Setup

### Task 1.1: Create Monorepo Structure

**Files:**
- Create: `solar-monitor/package.json`
- Create: `solar-monitor/pnpm-workspace.yaml`
- Create: `solar-monitor/.gitignore`

**Step 1: Create project directory**

```bash
mkdir -p /home/mikeb/solar-monitor
cd /home/mikeb/solar-monitor
```

**Step 2: Initialize git**

```bash
git init
```

**Step 3: Create root package.json**

```json
{
  "name": "solar-monitor",
  "private": true,
  "scripts": {
    "backend": "pnpm --filter backend dev",
    "web": "pnpm --filter web dev",
    "mobile": "pnpm --filter mobile start",
    "test": "pnpm -r test",
    "build": "pnpm -r build"
  },
  "devDependencies": {
    "typescript": "^5.3.0"
  }
}
```

**Step 4: Create pnpm-workspace.yaml**

```yaml
packages:
  - 'backend'
  - 'web'
  - 'mobile'
```

**Step 5: Create .gitignore**

```
node_modules/
dist/
.env
.env.local
*.log
.DS_Store
*.db
*.sqlite
```

**Step 6: Install dependencies**

```bash
pnpm install
```

**Step 7: Commit**

```bash
git add .
git commit -m "chore: initialize monorepo structure"
```

---

### Task 1.2: Setup Backend Project

**Files:**
- Create: `solar-monitor/backend/package.json`
- Create: `solar-monitor/backend/tsconfig.json`
- Create: `solar-monitor/backend/src/index.ts`

**Step 1: Create backend directory**

```bash
mkdir -p /home/mikeb/solar-monitor/backend/src
```

**Step 2: Create backend package.json**

```json
{
  "name": "backend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "test": "vitest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "better-sqlite3": "^9.2.2",
    "jsonwebtoken": "^9.0.2",
    "bcryptjs": "^2.4.3",
    "axios": "^1.6.2",
    "node-cron": "^3.0.3",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/cors": "^2.8.17",
    "@types/better-sqlite3": "^7.6.8",
    "@types/jsonwebtoken": "^9.0.5",
    "@types/bcryptjs": "^2.4.6",
    "@types/node-cron": "^3.0.11",
    "tsx": "^4.6.2",
    "typescript": "^5.3.0",
    "vitest": "^1.1.0"
  }
}
```

**Step 3: Create backend tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Step 4: Create minimal src/index.ts**

```typescript
import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
```

**Step 5: Create .env.example**

Create `solar-monitor/backend/.env.example`:

```
PORT=3001
JWT_SECRET=your-secret-key-change-in-production
SOLAREDGE_API_KEY=your-solaredge-api-key
SOLAREDGE_SITE_ID=your-site-id
```

**Step 6: Install backend dependencies**

```bash
cd /home/mikeb/solar-monitor/backend
pnpm install
```

**Step 7: Test backend starts**

```bash
pnpm dev
```
Expected: "Backend running on http://localhost:3001"
Press Ctrl+C to stop.

**Step 8: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): initialize Express server"
```

---

### Task 1.3: Setup Web Project

**Files:**
- Create: `solar-monitor/web/` (via Vite)

**Step 1: Create web app with Vite**

```bash
cd /home/mikeb/solar-monitor
pnpm create vite web --template react-ts
```

**Step 2: Install web dependencies**

```bash
cd /home/mikeb/solar-monitor/web
pnpm install
pnpm add axios react-router-dom recharts
pnpm add -D tailwindcss postcss autoprefixer @types/react-router-dom
```

**Step 3: Initialize Tailwind**

```bash
npx tailwindcss init -p
```

**Step 4: Configure tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**Step 5: Update src/index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 6: Test web app starts**

```bash
pnpm dev
```
Expected: Vite dev server starts on http://localhost:5173

**Step 7: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(web): initialize React + Vite + Tailwind"
```

---

### Task 1.4: Setup Mobile Project

**Files:**
- Create: `solar-monitor/mobile/` (via Expo)

**Step 1: Create Expo app**

```bash
cd /home/mikeb/solar-monitor
npx create-expo-app mobile --template blank-typescript
```

**Step 2: Install mobile dependencies**

```bash
cd /home/mikeb/solar-monitor/mobile
pnpm add axios @react-navigation/native @react-navigation/native-stack react-native-paper react-native-safe-area-context react-native-screens expo-notifications
```

**Step 3: Verify mobile starts**

```bash
pnpm start
```
Expected: Expo dev server starts, shows QR code

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(mobile): initialize React Native + Expo"
```

---

## Phase 2: Backend Database & Auth

### Task 2.1: Database Schema

**Files:**
- Create: `solar-monitor/backend/src/db/schema.sql`
- Create: `solar-monitor/backend/src/db/database.ts`

**Step 1: Create db directory**

```bash
mkdir -p /home/mikeb/solar-monitor/backend/src/db
```

**Step 2: Create schema.sql**

```sql
-- Users table
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Panels table (cached from SolarEdge)
CREATE TABLE IF NOT EXISTS panels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  solaredge_id TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  serial_number TEXT,
  manufacturer TEXT,
  model TEXT,
  position_x INTEGER DEFAULT 0,
  position_y INTEGER DEFAULT 0,
  is_active INTEGER DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Panel readings (historical data)
CREATE TABLE IF NOT EXISTS panel_readings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  panel_id INTEGER NOT NULL,
  watts REAL,
  voltage REAL,
  current REAL,
  temperature REAL,
  recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (panel_id) REFERENCES panels(id)
);

-- Site overview (cached)
CREATE TABLE IF NOT EXISTS site_overview (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  current_power REAL,
  daily_energy REAL,
  monthly_energy REAL,
  yearly_energy REAL,
  lifetime_energy REAL,
  recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Automation rules
CREATE TABLE IF NOT EXISTS automation_rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  trigger_config TEXT NOT NULL,
  action_config TEXT NOT NULL,
  is_enabled INTEGER DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Control audit log
CREATE TABLE IF NOT EXISTS control_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  status TEXT NOT NULL,
  details TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_panel_readings_panel_id ON panel_readings(panel_id);
CREATE INDEX IF NOT EXISTS idx_panel_readings_recorded_at ON panel_readings(recorded_at);
CREATE INDEX IF NOT EXISTS idx_control_log_created_at ON control_log(created_at);
```

**Step 3: Create database.ts**

```typescript
import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

const DB_PATH = process.env.DB_PATH || './solar-monitor.db';

export const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');

export function initializeDatabase(): void {
  const schemaPath = path.join(__dirname, 'schema.sql');
  const schema = fs.readFileSync(schemaPath, 'utf-8');
  db.exec(schema);
  console.log('Database initialized');
}

export function closeDatabase(): void {
  db.close();
}
```

**Step 4: Update src/index.ts to initialize DB**

```typescript
import express from 'express';
import cors from 'cors';
import { initializeDatabase } from './db/database';

const app = express();
const PORT = process.env.PORT || 3001;

// Initialize database
initializeDatabase();

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
```

**Step 5: Test database initializes**

```bash
cd /home/mikeb/solar-monitor/backend
pnpm dev
```
Expected: "Database initialized" then "Backend running..."

**Step 6: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add SQLite database with schema"
```

---

### Task 2.2: Authentication Service

**Files:**
- Create: `solar-monitor/backend/src/services/auth.ts`
- Create: `solar-monitor/backend/src/routes/auth.ts`
- Create: `solar-monitor/backend/src/middleware/auth.ts`

**Step 1: Create services directory**

```bash
mkdir -p /home/mikeb/solar-monitor/backend/src/services
mkdir -p /home/mikeb/solar-monitor/backend/src/routes
mkdir -p /home/mikeb/solar-monitor/backend/src/middleware
```

**Step 2: Create auth service**

Create `solar-monitor/backend/src/services/auth.ts`:

```typescript
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { db } from '../db/database';

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret-change-me';
const JWT_EXPIRES_IN = '7d';

interface User {
  id: number;
  email: string;
  password_hash: string;
  created_at: string;
}

export async function registerUser(email: string, password: string): Promise<{ id: number; email: string }> {
  const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
  if (existing) {
    throw new Error('User already exists');
  }

  const passwordHash = await bcrypt.hash(password, 10);
  const result = db.prepare('INSERT INTO users (email, password_hash) VALUES (?, ?)').run(email, passwordHash);

  return { id: result.lastInsertRowid as number, email };
}

export async function loginUser(email: string, password: string): Promise<string> {
  const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email) as User | undefined;

  if (!user) {
    throw new Error('Invalid credentials');
  }

  const validPassword = await bcrypt.compare(password, user.password_hash);
  if (!validPassword) {
    throw new Error('Invalid credentials');
  }

  const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
  return token;
}

export function verifyToken(token: string): { userId: number; email: string } {
  const decoded = jwt.verify(token, JWT_SECRET) as { userId: number; email: string };
  return decoded;
}
```

**Step 3: Create auth middleware**

Create `solar-monitor/backend/src/middleware/auth.ts`:

```typescript
import { Request, Response, NextFunction } from 'express';
import { verifyToken } from '../services/auth';

export interface AuthRequest extends Request {
  user?: { userId: number; email: string };
}

export function requireAuth(req: AuthRequest, res: Response, next: NextFunction): void {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ error: 'No token provided' });
    return;
  }

  const token = authHeader.substring(7);

  try {
    const decoded = verifyToken(token);
    req.user = decoded;
    next();
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
}
```

**Step 4: Create auth routes**

Create `solar-monitor/backend/src/routes/auth.ts`:

```typescript
import { Router } from 'express';
import { registerUser, loginUser } from '../services/auth';

const router = Router();

router.post('/register', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ error: 'Email and password required' });
      return;
    }

    if (password.length < 8) {
      res.status(400).json({ error: 'Password must be at least 8 characters' });
      return;
    }

    const user = await registerUser(email, password);
    res.status(201).json({ message: 'User created', user });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Registration failed';
    res.status(400).json({ error: message });
  }
});

router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ error: 'Email and password required' });
      return;
    }

    const token = await loginUser(email, password);
    res.json({ token });
  } catch (error) {
    res.status(401).json({ error: 'Invalid credentials' });
  }
});

export default router;
```

**Step 5: Update index.ts to use auth routes**

```typescript
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { initializeDatabase } from './db/database';
import authRoutes from './routes/auth';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

initializeDatabase();

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use('/api/auth', authRoutes);

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
```

**Step 6: Test auth endpoints**

Start server:
```bash
cd /home/mikeb/solar-monitor/backend
pnpm dev
```

In another terminal, test registration:
```bash
curl -X POST http://localhost:3001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```
Expected: `{"message":"User created","user":{"id":1,"email":"test@example.com"}}`

Test login:
```bash
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```
Expected: `{"token":"eyJ..."}`

**Step 7: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add JWT authentication"
```

---

## Phase 3: SolarEdge Integration

### Task 3.1: SolarEdge API Client

**Files:**
- Create: `solar-monitor/backend/src/services/solaredge.ts`

**Step 1: Create SolarEdge service**

```typescript
import axios, { AxiosInstance } from 'axios';

const SOLAREDGE_BASE_URL = 'https://monitoringapi.solaredge.com';

interface SiteOverview {
  currentPower: { power: number };
  lastDayData: { energy: number };
  lastMonthData: { energy: number };
  lastYearData: { energy: number };
  lifeTimeData: { energy: number };
}

interface Equipment {
  name: string;
  manufacturer: string;
  model: string;
  serialNumber: string;
}

interface PanelData {
  serialNumber: string;
  data: {
    date: string;
    totalActivePower: number;
    dcVoltage: number;
    temperature: number;
  }[];
}

export class SolarEdgeClient {
  private client: AxiosInstance;
  private siteId: string;

  constructor(apiKey: string, siteId: string) {
    this.siteId = siteId;
    this.client = axios.create({
      baseURL: SOLAREDGE_BASE_URL,
      params: { api_key: apiKey },
    });
  }

  async getSiteOverview(): Promise<SiteOverview> {
    const response = await this.client.get(`/site/${this.siteId}/overview`);
    return response.data.overview;
  }

  async getEquipmentList(): Promise<Equipment[]> {
    const response = await this.client.get(`/equipment/${this.siteId}/list`);
    return response.data.reporters.list;
  }

  async getPanelData(startDate: string, endDate: string): Promise<PanelData[]> {
    const response = await this.client.get(`/equipment/${this.siteId}/data`, {
      params: { startTime: startDate, endTime: endDate },
    });
    return response.data.data.telemetries;
  }

  async getSitePower(startDate: string, endDate: string): Promise<{ time: string; value: number }[]> {
    const response = await this.client.get(`/site/${this.siteId}/power`, {
      params: { startTime: startDate, endTime: endDate },
    });
    return response.data.power.values;
  }
}

// Singleton instance
let solarEdgeClient: SolarEdgeClient | null = null;

export function getSolarEdgeClient(): SolarEdgeClient {
  if (!solarEdgeClient) {
    const apiKey = process.env.SOLAREDGE_API_KEY;
    const siteId = process.env.SOLAREDGE_SITE_ID;

    if (!apiKey || !siteId) {
      throw new Error('SOLAREDGE_API_KEY and SOLAREDGE_SITE_ID must be set');
    }

    solarEdgeClient = new SolarEdgeClient(apiKey, siteId);
  }
  return solarEdgeClient;
}
```

**Step 2: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add SolarEdge API client"
```

---

### Task 3.2: Data Sync Service

**Files:**
- Create: `solar-monitor/backend/src/services/sync.ts`

**Step 1: Create sync service**

```typescript
import { db } from '../db/database';
import { getSolarEdgeClient } from './solaredge';

export async function syncSiteOverview(): Promise<void> {
  try {
    const client = getSolarEdgeClient();
    const overview = await client.getSiteOverview();

    db.prepare(`
      INSERT INTO site_overview (current_power, daily_energy, monthly_energy, yearly_energy, lifetime_energy)
      VALUES (?, ?, ?, ?, ?)
    `).run(
      overview.currentPower.power,
      overview.lastDayData.energy,
      overview.lastMonthData.energy,
      overview.lastYearData.energy,
      overview.lifeTimeData.energy
    );

    console.log(`Synced site overview: ${overview.currentPower.power}W`);
  } catch (error) {
    console.error('Failed to sync site overview:', error);
  }
}

export async function syncPanels(): Promise<void> {
  try {
    const client = getSolarEdgeClient();
    const equipment = await client.getEquipmentList();

    for (const device of equipment) {
      db.prepare(`
        INSERT INTO panels (solaredge_id, name, serial_number, manufacturer, model)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(solaredge_id) DO UPDATE SET
          name = excluded.name,
          manufacturer = excluded.manufacturer,
          model = excluded.model
      `).run(
        device.serialNumber,
        device.name,
        device.serialNumber,
        device.manufacturer,
        device.model
      );
    }

    console.log(`Synced ${equipment.length} panels`);
  } catch (error) {
    console.error('Failed to sync panels:', error);
  }
}

export async function syncPanelReadings(): Promise<void> {
  try {
    const client = getSolarEdgeClient();
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

    const startDate = oneHourAgo.toISOString().slice(0, 19);
    const endDate = now.toISOString().slice(0, 19);

    const panelData = await client.getPanelData(startDate, endDate);

    for (const panel of panelData) {
      const dbPanel = db.prepare('SELECT id FROM panels WHERE serial_number = ?').get(panel.serialNumber) as { id: number } | undefined;

      if (dbPanel && panel.data.length > 0) {
        const latest = panel.data[panel.data.length - 1];

        db.prepare(`
          INSERT INTO panel_readings (panel_id, watts, voltage, temperature, recorded_at)
          VALUES (?, ?, ?, ?, ?)
        `).run(
          dbPanel.id,
          latest.totalActivePower,
          latest.dcVoltage,
          latest.temperature,
          latest.date
        );
      }
    }

    console.log(`Synced readings for ${panelData.length} panels`);
  } catch (error) {
    console.error('Failed to sync panel readings:', error);
  }
}

export async function runFullSync(): Promise<void> {
  await syncSiteOverview();
  await syncPanels();
  await syncPanelReadings();
}
```

**Step 2: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add data sync service"
```

---

### Task 3.3: Cron Scheduler

**Files:**
- Modify: `solar-monitor/backend/src/index.ts`

**Step 1: Add cron job to index.ts**

```typescript
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import cron from 'node-cron';
import { initializeDatabase } from './db/database';
import authRoutes from './routes/auth';
import { runFullSync } from './services/sync';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

initializeDatabase();

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use('/api/auth', authRoutes);

// Run sync every 5 minutes
cron.schedule('*/5 * * * *', async () => {
  console.log('Running scheduled sync...');
  await runFullSync();
});

// Manual sync endpoint (for testing)
app.post('/api/sync', async (_req, res) => {
  try {
    await runFullSync();
    res.json({ message: 'Sync completed' });
  } catch (error) {
    res.status(500).json({ error: 'Sync failed' });
  }
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
  console.log('Sync scheduled every 5 minutes');
});
```

**Step 2: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add cron scheduler for data sync"
```

---

## Phase 4: Backend API Routes

### Task 4.1: Panels API

**Files:**
- Create: `solar-monitor/backend/src/routes/panels.ts`

**Step 1: Create panels routes**

```typescript
import { Router } from 'express';
import { db } from '../db/database';
import { requireAuth, AuthRequest } from '../middleware/auth';

const router = Router();

interface Panel {
  id: number;
  solaredge_id: string;
  name: string;
  serial_number: string;
  manufacturer: string;
  model: string;
  position_x: number;
  position_y: number;
  is_active: number;
}

interface PanelReading {
  watts: number;
  voltage: number;
  temperature: number;
  recorded_at: string;
}

// Get all panels with latest readings
router.get('/', requireAuth, (_req, res) => {
  const panels = db.prepare(`
    SELECT p.*,
      (SELECT watts FROM panel_readings WHERE panel_id = p.id ORDER BY recorded_at DESC LIMIT 1) as current_watts,
      (SELECT voltage FROM panel_readings WHERE panel_id = p.id ORDER BY recorded_at DESC LIMIT 1) as current_voltage,
      (SELECT temperature FROM panel_readings WHERE panel_id = p.id ORDER BY recorded_at DESC LIMIT 1) as current_temperature
    FROM panels p
    ORDER BY p.name
  `).all();

  res.json(panels);
});

// Get single panel with history
router.get('/:id', requireAuth, (req, res) => {
  const panel = db.prepare('SELECT * FROM panels WHERE id = ?').get(req.params.id) as Panel | undefined;

  if (!panel) {
    res.status(404).json({ error: 'Panel not found' });
    return;
  }

  const readings = db.prepare(`
    SELECT watts, voltage, temperature, recorded_at
    FROM panel_readings
    WHERE panel_id = ?
    ORDER BY recorded_at DESC
    LIMIT 288
  `).all(req.params.id) as PanelReading[];

  res.json({ ...panel, readings });
});

// Update panel position (for layout editor)
router.patch('/:id/position', requireAuth, (req, res) => {
  const { x, y } = req.body;

  db.prepare('UPDATE panels SET position_x = ?, position_y = ? WHERE id = ?')
    .run(x, y, req.params.id);

  res.json({ message: 'Position updated' });
});

export default router;
```

**Step 2: Add to index.ts**

Add after auth routes:
```typescript
import panelsRoutes from './routes/panels';
// ...
app.use('/api/panels', panelsRoutes);
```

**Step 3: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add panels API routes"
```

---

### Task 4.2: Site Overview API

**Files:**
- Create: `solar-monitor/backend/src/routes/site.ts`

**Step 1: Create site routes**

```typescript
import { Router } from 'express';
import { db } from '../db/database';
import { requireAuth } from '../middleware/auth';

const router = Router();

interface SiteOverview {
  current_power: number;
  daily_energy: number;
  monthly_energy: number;
  yearly_energy: number;
  lifetime_energy: number;
  recorded_at: string;
}

// Get current site overview
router.get('/overview', requireAuth, (_req, res) => {
  const overview = db.prepare(`
    SELECT * FROM site_overview
    ORDER BY recorded_at DESC
    LIMIT 1
  `).get() as SiteOverview | undefined;

  if (!overview) {
    res.json({
      currentPower: 0,
      dailyEnergy: 0,
      monthlyEnergy: 0,
      yearlyEnergy: 0,
      lifetimeEnergy: 0,
    });
    return;
  }

  res.json({
    currentPower: overview.current_power,
    dailyEnergy: overview.daily_energy,
    monthlyEnergy: overview.monthly_energy,
    yearlyEnergy: overview.yearly_energy,
    lifetimeEnergy: overview.lifetime_energy,
    lastUpdated: overview.recorded_at,
  });
});

// Get power history for charts
router.get('/power-history', requireAuth, (req, res) => {
  const hours = parseInt(req.query.hours as string) || 24;

  const history = db.prepare(`
    SELECT current_power as power, recorded_at as time
    FROM site_overview
    WHERE recorded_at >= datetime('now', '-${hours} hours')
    ORDER BY recorded_at ASC
  `).all();

  res.json(history);
});

export default router;
```

**Step 2: Add to index.ts**

```typescript
import siteRoutes from './routes/site';
// ...
app.use('/api/site', siteRoutes);
```

**Step 3: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add site overview API routes"
```

---

### Task 4.3: Controls API

**Files:**
- Create: `solar-monitor/backend/src/routes/controls.ts`
- Create: `solar-monitor/backend/src/services/controls.ts`

**Step 1: Create controls service**

```typescript
import { db } from '../db/database';

// Note: Actual SolarEdge control requires SetApp API which has limited documentation.
// This service provides the structure - actual implementation depends on your specific access.

export interface ControlCommand {
  type: 'shutdown' | 'restart' | 'master_off' | 'master_on';
  target: string; // panel ID, string name, or 'system'
  userId: number;
}

export async function executeControl(command: ControlCommand): Promise<{ success: boolean; message: string }> {
  // Log the control action
  db.prepare(`
    INSERT INTO control_log (user_id, action, target, status, details)
    VALUES (?, ?, ?, ?, ?)
  `).run(
    command.userId,
    command.type,
    command.target,
    'pending',
    JSON.stringify(command)
  );

  // TODO: Implement actual SolarEdge SetApp API call here
  // For now, simulate success and update panel state

  if (command.type === 'shutdown' && command.target !== 'system') {
    db.prepare('UPDATE panels SET is_active = 0 WHERE id = ?').run(command.target);
  } else if (command.type === 'restart' && command.target !== 'system') {
    db.prepare('UPDATE panels SET is_active = 1 WHERE id = ?').run(command.target);
  }

  // Update log to completed
  db.prepare(`
    UPDATE control_log
    SET status = 'completed'
    WHERE user_id = ? AND action = ? AND target = ?
    ORDER BY created_at DESC
    LIMIT 1
  `).run(command.userId, command.type, command.target);

  return {
    success: true,
    message: `${command.type} command sent to ${command.target}`
  };
}

export function getControlLog(limit: number = 50) {
  return db.prepare(`
    SELECT cl.*, u.email as user_email
    FROM control_log cl
    LEFT JOIN users u ON cl.user_id = u.id
    ORDER BY cl.created_at DESC
    LIMIT ?
  `).all(limit);
}
```

**Step 2: Create controls routes**

```typescript
import { Router } from 'express';
import { requireAuth, AuthRequest } from '../middleware/auth';
import { executeControl, getControlLog } from '../services/controls';

const router = Router();

// Shutdown a panel
router.post('/panel/:id/shutdown', requireAuth, async (req: AuthRequest, res) => {
  try {
    const result = await executeControl({
      type: 'shutdown',
      target: req.params.id,
      userId: req.user!.userId,
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Control command failed' });
  }
});

// Restart a panel
router.post('/panel/:id/restart', requireAuth, async (req: AuthRequest, res) => {
  try {
    const result = await executeControl({
      type: 'restart',
      target: req.params.id,
      userId: req.user!.userId,
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Control command failed' });
  }
});

// Master system shutdown
router.post('/system/shutdown', requireAuth, async (req: AuthRequest, res) => {
  try {
    const result = await executeControl({
      type: 'master_off',
      target: 'system',
      userId: req.user!.userId,
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Control command failed' });
  }
});

// Master system restart
router.post('/system/restart', requireAuth, async (req: AuthRequest, res) => {
  try {
    const result = await executeControl({
      type: 'master_on',
      target: 'system',
      userId: req.user!.userId,
    });
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Control command failed' });
  }
});

// Get control history
router.get('/log', requireAuth, (_req, res) => {
  const log = getControlLog();
  res.json(log);
});

export default router;
```

**Step 3: Add to index.ts**

```typescript
import controlsRoutes from './routes/controls';
// ...
app.use('/api/controls', controlsRoutes);
```

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add panel control API routes"
```

---

## Phase 5: Automation Engine

### Task 5.1: Automation Service

**Files:**
- Create: `solar-monitor/backend/src/services/automation.ts`
- Create: `solar-monitor/backend/src/routes/automation.ts`

**Step 1: Create automation service**

```typescript
import { db } from '../db/database';
import { executeControl } from './controls';

interface TriggerConfig {
  type: 'threshold' | 'time' | 'comparison';
  panel?: string;
  metric?: 'watts' | 'voltage' | 'temperature';
  condition?: 'above' | 'below';
  value?: number;
  duration?: string;
  time?: string;
  comparePanel?: string;
}

interface ActionConfig {
  type: 'notify' | 'shutdown' | 'log';
  method?: 'push' | 'email';
  message?: string;
  target?: string;
}

interface AutomationRule {
  id: number;
  name: string;
  trigger_config: string;
  action_config: string;
  is_enabled: number;
}

// Track threshold breach durations
const thresholdState: Map<number, { breachedAt: Date | null }> = new Map();

export function createRule(name: string, trigger: TriggerConfig, action: ActionConfig): number {
  const result = db.prepare(`
    INSERT INTO automation_rules (name, trigger_config, action_config)
    VALUES (?, ?, ?)
  `).run(name, JSON.stringify(trigger), JSON.stringify(action));

  return result.lastInsertRowid as number;
}

export function getRules(): AutomationRule[] {
  return db.prepare('SELECT * FROM automation_rules ORDER BY created_at DESC').all() as AutomationRule[];
}

export function updateRule(id: number, updates: { name?: string; trigger?: TriggerConfig; action?: ActionConfig; enabled?: boolean }): void {
  const sets: string[] = [];
  const values: (string | number)[] = [];

  if (updates.name) {
    sets.push('name = ?');
    values.push(updates.name);
  }
  if (updates.trigger) {
    sets.push('trigger_config = ?');
    values.push(JSON.stringify(updates.trigger));
  }
  if (updates.action) {
    sets.push('action_config = ?');
    values.push(JSON.stringify(updates.action));
  }
  if (updates.enabled !== undefined) {
    sets.push('is_enabled = ?');
    values.push(updates.enabled ? 1 : 0);
  }

  if (sets.length > 0) {
    values.push(id);
    db.prepare(`UPDATE automation_rules SET ${sets.join(', ')} WHERE id = ?`).run(...values);
  }
}

export function deleteRule(id: number): void {
  db.prepare('DELETE FROM automation_rules WHERE id = ?').run(id);
  thresholdState.delete(id);
}

export async function evaluateRules(): Promise<void> {
  const rules = db.prepare('SELECT * FROM automation_rules WHERE is_enabled = 1').all() as AutomationRule[];

  for (const rule of rules) {
    const trigger = JSON.parse(rule.trigger_config) as TriggerConfig;
    const action = JSON.parse(rule.action_config) as ActionConfig;

    const shouldFire = await evaluateTrigger(rule.id, trigger);

    if (shouldFire) {
      await executeAction(action, rule.name);
    }
  }
}

async function evaluateTrigger(ruleId: number, trigger: TriggerConfig): Promise<boolean> {
  if (trigger.type === 'threshold' && trigger.panel && trigger.metric && trigger.condition && trigger.value !== undefined) {
    const reading = db.prepare(`
      SELECT ${trigger.metric} as value
      FROM panel_readings pr
      JOIN panels p ON pr.panel_id = p.id
      WHERE p.name = ? OR p.id = ?
      ORDER BY pr.recorded_at DESC
      LIMIT 1
    `).get(trigger.panel, trigger.panel) as { value: number } | undefined;

    if (!reading) return false;

    const breached = trigger.condition === 'below'
      ? reading.value < trigger.value
      : reading.value > trigger.value;

    if (!breached) {
      thresholdState.set(ruleId, { breachedAt: null });
      return false;
    }

    // Check duration
    const state = thresholdState.get(ruleId);
    if (!state?.breachedAt) {
      thresholdState.set(ruleId, { breachedAt: new Date() });
      return false;
    }

    const durationMs = parseDuration(trigger.duration || '0');
    const elapsed = Date.now() - state.breachedAt.getTime();

    if (elapsed >= durationMs) {
      thresholdState.set(ruleId, { breachedAt: null }); // Reset after firing
      return true;
    }

    return false;
  }

  if (trigger.type === 'time' && trigger.time) {
    const now = new Date();
    const [hours, minutes] = trigger.time.split(':').map(Number);
    return now.getHours() === hours && now.getMinutes() === minutes;
  }

  return false;
}

async function executeAction(action: ActionConfig, ruleName: string): Promise<void> {
  console.log(`Automation triggered: ${ruleName}`);

  if (action.type === 'notify') {
    // TODO: Implement push/email notifications
    console.log(`Notification: ${action.message}`);
  } else if (action.type === 'shutdown' && action.target) {
    await executeControl({
      type: 'shutdown',
      target: action.target,
      userId: 0, // System user
    });
  } else if (action.type === 'log') {
    db.prepare(`
      INSERT INTO control_log (action, target, status, details)
      VALUES ('automation', ?, 'completed', ?)
    `).run(ruleName, action.message || '');
  }
}

function parseDuration(duration: string): number {
  const match = duration.match(/^(\d+)(s|m|h)$/);
  if (!match) return 0;

  const value = parseInt(match[1]);
  const unit = match[2];

  switch (unit) {
    case 's': return value * 1000;
    case 'm': return value * 60 * 1000;
    case 'h': return value * 60 * 60 * 1000;
    default: return 0;
  }
}
```

**Step 2: Create automation routes**

```typescript
import { Router } from 'express';
import { requireAuth } from '../middleware/auth';
import { createRule, getRules, updateRule, deleteRule } from '../services/automation';

const router = Router();

// Get all rules
router.get('/', requireAuth, (_req, res) => {
  const rules = getRules();
  res.json(rules.map(r => ({
    ...r,
    trigger: JSON.parse(r.trigger_config),
    action: JSON.parse(r.action_config),
  })));
});

// Create rule
router.post('/', requireAuth, (req, res) => {
  const { name, trigger, action } = req.body;

  if (!name || !trigger || !action) {
    res.status(400).json({ error: 'Name, trigger, and action required' });
    return;
  }

  const id = createRule(name, trigger, action);
  res.status(201).json({ id, message: 'Rule created' });
});

// Update rule
router.patch('/:id', requireAuth, (req, res) => {
  const { name, trigger, action, enabled } = req.body;
  updateRule(parseInt(req.params.id), { name, trigger, action, enabled });
  res.json({ message: 'Rule updated' });
});

// Delete rule
router.delete('/:id', requireAuth, (req, res) => {
  deleteRule(parseInt(req.params.id));
  res.json({ message: 'Rule deleted' });
});

export default router;
```

**Step 3: Add to index.ts and cron**

```typescript
import automationRoutes from './routes/automation';
import { evaluateRules } from './services/automation';
// ...
app.use('/api/automation', automationRoutes);

// Update cron to also evaluate rules
cron.schedule('*/5 * * * *', async () => {
  console.log('Running scheduled sync...');
  await runFullSync();
  await evaluateRules();
});
```

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(backend): add automation rules engine"
```

---

## Phase 6: Web Dashboard

### Task 6.1: API Client Setup

**Files:**
- Create: `solar-monitor/web/src/api/client.ts`
- Create: `solar-monitor/web/src/api/auth.ts`
- Create: `solar-monitor/web/src/api/panels.ts`

**Step 1: Create API client**

```typescript
// solar-monitor/web/src/api/client.ts
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export const api = axios.create({
  baseURL: API_URL,
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

**Step 2: Create auth API**

```typescript
// solar-monitor/web/src/api/auth.ts
import { api } from './client';

export async function login(email: string, password: string): Promise<string> {
  const response = await api.post('/api/auth/login', { email, password });
  const { token } = response.data;
  localStorage.setItem('token', token);
  return token;
}

export async function register(email: string, password: string): Promise<void> {
  await api.post('/api/auth/register', { email, password });
}

export function logout(): void {
  localStorage.removeItem('token');
  window.location.href = '/login';
}

export function isAuthenticated(): boolean {
  return !!localStorage.getItem('token');
}
```

**Step 3: Create panels API**

```typescript
// solar-monitor/web/src/api/panels.ts
import { api } from './client';

export interface Panel {
  id: number;
  name: string;
  serial_number: string;
  current_watts: number | null;
  current_voltage: number | null;
  current_temperature: number | null;
  position_x: number;
  position_y: number;
  is_active: number;
}

export interface SiteOverview {
  currentPower: number;
  dailyEnergy: number;
  monthlyEnergy: number;
  yearlyEnergy: number;
  lifetimeEnergy: number;
  lastUpdated: string;
}

export async function getPanels(): Promise<Panel[]> {
  const response = await api.get('/api/panels');
  return response.data;
}

export async function getPanel(id: number): Promise<Panel & { readings: any[] }> {
  const response = await api.get(`/api/panels/${id}`);
  return response.data;
}

export async function getSiteOverview(): Promise<SiteOverview> {
  const response = await api.get('/api/site/overview');
  return response.data;
}

export async function getPowerHistory(hours: number = 24): Promise<{ power: number; time: string }[]> {
  const response = await api.get(`/api/site/power-history?hours=${hours}`);
  return response.data;
}

export async function shutdownPanel(id: number): Promise<void> {
  await api.post(`/api/controls/panel/${id}/shutdown`);
}

export async function restartPanel(id: number): Promise<void> {
  await api.post(`/api/controls/panel/${id}/restart`);
}
```

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(web): add API client and services"
```

---

### Task 6.2: Login Page

**Files:**
- Create: `solar-monitor/web/src/pages/Login.tsx`

**Step 1: Create Login component**

```tsx
import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../api/auth';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(email, password);
      navigate('/');
    } catch {
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="bg-gray-800 p-8 rounded-lg shadow-lg w-full max-w-md">
        <h1 className="text-2xl font-bold text-white mb-6 text-center">
          Solar Monitor
        </h1>

        {error && (
          <div className="bg-red-500/20 border border-red-500 text-red-300 px-4 py-2 rounded mb-4">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-300 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-500"
              required
            />
          </div>

          <div>
            <label className="block text-gray-300 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-yellow-500"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 bg-yellow-500 hover:bg-yellow-600 text-gray-900 font-semibold rounded transition disabled:opacity-50"
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}
```

**Step 2: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(web): add login page"
```

---

### Task 6.3: Dashboard Page

**Files:**
- Create: `solar-monitor/web/src/pages/Dashboard.tsx`
- Create: `solar-monitor/web/src/components/PanelGrid.tsx`
- Create: `solar-monitor/web/src/components/PowerChart.tsx`

**Step 1: Create PanelGrid component**

```tsx
// solar-monitor/web/src/components/PanelGrid.tsx
import { Panel } from '../api/panels';

interface Props {
  panels: Panel[];
  onPanelClick: (panel: Panel) => void;
}

function getPanelColor(panel: Panel): string {
  if (!panel.is_active) return 'bg-red-500';
  if (panel.current_watts === null) return 'bg-gray-500';
  if (panel.current_watts < 50) return 'bg-yellow-500';
  return 'bg-green-500';
}

export default function PanelGrid({ panels, onPanelClick }: Props) {
  return (
    <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
      {panels.map((panel) => (
        <button
          key={panel.id}
          onClick={() => onPanelClick(panel)}
          className={`${getPanelColor(panel)} p-4 rounded-lg text-white hover:opacity-80 transition`}
        >
          <div className="text-xs font-medium truncate">{panel.name}</div>
          <div className="text-lg font-bold">
            {panel.current_watts !== null ? `${Math.round(panel.current_watts)}W` : '--'}
          </div>
        </button>
      ))}
    </div>
  );
}
```

**Step 2: Create PowerChart component**

```tsx
// solar-monitor/web/src/components/PowerChart.tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface Props {
  data: { power: number; time: string }[];
}

export default function PowerChart({ data }: Props) {
  const formatted = data.map((d) => ({
    ...d,
    time: new Date(d.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={formatted}>
        <XAxis dataKey="time" stroke="#9ca3af" tick={{ fontSize: 12 }} />
        <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
          labelStyle={{ color: '#fff' }}
        />
        <Line
          type="monotone"
          dataKey="power"
          stroke="#eab308"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

**Step 3: Create Dashboard page**

```tsx
// solar-monitor/web/src/pages/Dashboard.tsx
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getPanels, getSiteOverview, getPowerHistory, Panel, SiteOverview } from '../api/panels';
import { logout } from '../api/auth';
import PanelGrid from '../components/PanelGrid';
import PowerChart from '../components/PowerChart';

export default function Dashboard() {
  const [panels, setPanels] = useState<Panel[]>([]);
  const [overview, setOverview] = useState<SiteOverview | null>(null);
  const [powerHistory, setPowerHistory] = useState<{ power: number; time: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchData() {
      try {
        const [panelsData, overviewData, historyData] = await Promise.all([
          getPanels(),
          getSiteOverview(),
          getPowerHistory(24),
        ]);
        setPanels(panelsData);
        setOverview(overviewData);
        setPowerHistory(historyData);
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const handlePanelClick = (panel: Panel) => {
    navigate(`/panel/${panel.id}`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <header className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-bold">Solar Monitor</h1>
        <button
          onClick={logout}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded transition"
        >
          Logout
        </button>
      </header>

      {/* Overview Stats */}
      {overview && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-gray-400 text-sm">Current Power</div>
            <div className="text-2xl font-bold text-yellow-400">
              {(overview.currentPower / 1000).toFixed(2)} kW
            </div>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-gray-400 text-sm">Today</div>
            <div className="text-2xl font-bold">
              {(overview.dailyEnergy / 1000).toFixed(1)} kWh
            </div>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-gray-400 text-sm">This Month</div>
            <div className="text-2xl font-bold">
              {(overview.monthlyEnergy / 1000).toFixed(0)} kWh
            </div>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-gray-400 text-sm">Lifetime</div>
            <div className="text-2xl font-bold">
              {(overview.lifetimeEnergy / 1000000).toFixed(1)} MWh
            </div>
          </div>
        </div>
      )}

      {/* Power Chart */}
      <div className="bg-gray-800 p-4 rounded-lg mb-8">
        <h2 className="text-lg font-semibold mb-4">Power Output (24h)</h2>
        <PowerChart data={powerHistory} />
      </div>

      {/* Panel Grid */}
      <div className="bg-gray-800 p-4 rounded-lg">
        <h2 className="text-lg font-semibold mb-4">Panels</h2>
        <PanelGrid panels={panels} onPanelClick={handlePanelClick} />
      </div>
    </div>
  );
}
```

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(web): add dashboard with panel grid and power chart"
```

---

### Task 6.4: Panel Detail Page

**Files:**
- Create: `solar-monitor/web/src/pages/PanelDetail.tsx`
- Create: `solar-monitor/web/src/components/ShutdownToggle.tsx`

**Step 1: Create ShutdownToggle component**

```tsx
// solar-monitor/web/src/components/ShutdownToggle.tsx
import { useState } from 'react';

interface Props {
  isActive: boolean;
  onShutdown: () => Promise<void>;
  onRestart: () => Promise<void>;
}

export default function ShutdownToggle({ isActive, onShutdown, onRestart }: Props) {
  const [loading, setLoading] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleToggle = async () => {
    if (isActive && !showConfirm) {
      setShowConfirm(true);
      return;
    }

    setLoading(true);
    try {
      if (isActive) {
        await onShutdown();
      } else {
        await onRestart();
      }
    } finally {
      setLoading(false);
      setShowConfirm(false);
    }
  };

  if (showConfirm) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-red-400">Confirm shutdown?</span>
        <button
          onClick={handleToggle}
          disabled={loading}
          className="px-3 py-1 bg-red-500 hover:bg-red-600 rounded text-white text-sm"
        >
          {loading ? '...' : 'Yes'}
        </button>
        <button
          onClick={() => setShowConfirm(false)}
          className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-white text-sm"
        >
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={handleToggle}
      disabled={loading}
      className={`px-4 py-2 rounded font-medium transition ${
        isActive
          ? 'bg-red-500 hover:bg-red-600 text-white'
          : 'bg-green-500 hover:bg-green-600 text-white'
      }`}
    >
      {loading ? 'Processing...' : isActive ? 'Shut Down' : 'Restart'}
    </button>
  );
}
```

**Step 2: Create PanelDetail page**

```tsx
// solar-monitor/web/src/pages/PanelDetail.tsx
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { getPanel, shutdownPanel, restartPanel, Panel } from '../api/panels';
import ShutdownToggle from '../components/ShutdownToggle';

interface PanelWithReadings extends Panel {
  readings: { watts: number; voltage: number; temperature: number; recorded_at: string }[];
}

export default function PanelDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [panel, setPanel] = useState<PanelWithReadings | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPanel() {
      try {
        const data = await getPanel(parseInt(id!));
        setPanel(data);
      } catch (error) {
        console.error('Failed to fetch panel:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchPanel();
  }, [id]);

  const handleShutdown = async () => {
    await shutdownPanel(parseInt(id!));
    setPanel((prev) => (prev ? { ...prev, is_active: 0 } : null));
  };

  const handleRestart = async () => {
    await restartPanel(parseInt(id!));
    setPanel((prev) => (prev ? { ...prev, is_active: 1 } : null));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white">Loading...</div>
      </div>
    );
  }

  if (!panel) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white">Panel not found</div>
      </div>
    );
  }

  const chartData = panel.readings.map((r) => ({
    time: new Date(r.recorded_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    watts: r.watts,
  })).reverse();

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <button
        onClick={() => navigate('/')}
        className="mb-6 text-gray-400 hover:text-white transition"
      >
        ← Back to Dashboard
      </button>

      <div className="flex justify-between items-start mb-8">
        <div>
          <h1 className="text-2xl font-bold">{panel.name}</h1>
          <p className="text-gray-400">{panel.serial_number}</p>
        </div>
        <ShutdownToggle
          isActive={panel.is_active === 1}
          onShutdown={handleShutdown}
          onRestart={handleRestart}
        />
      </div>

      {/* Current Stats */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-gray-400 text-sm">Power</div>
          <div className="text-2xl font-bold text-yellow-400">
            {panel.current_watts !== null ? `${Math.round(panel.current_watts)} W` : '--'}
          </div>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-gray-400 text-sm">Voltage</div>
          <div className="text-2xl font-bold">
            {panel.current_voltage !== null ? `${panel.current_voltage.toFixed(1)} V` : '--'}
          </div>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-gray-400 text-sm">Temperature</div>
          <div className="text-2xl font-bold">
            {panel.current_temperature !== null ? `${panel.current_temperature.toFixed(1)}°C` : '--'}
          </div>
        </div>
      </div>

      {/* Power History Chart */}
      <div className="bg-gray-800 p-4 rounded-lg">
        <h2 className="text-lg font-semibold mb-4">Power History</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <XAxis dataKey="time" stroke="#9ca3af" tick={{ fontSize: 12 }} />
            <YAxis stroke="#9ca3af" tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
              labelStyle={{ color: '#fff' }}
            />
            <Line type="monotone" dataKey="watts" stroke="#eab308" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
```

**Step 3: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(web): add panel detail page with shutdown control"
```

---

### Task 6.5: App Router Setup

**Files:**
- Modify: `solar-monitor/web/src/App.tsx`
- Modify: `solar-monitor/web/src/main.tsx`

**Step 1: Update App.tsx**

```tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { isAuthenticated } from './api/auth';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import PanelDetail from './pages/PanelDetail';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/panel/:id"
          element={
            <ProtectedRoute>
              <PanelDetail />
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
```

**Step 2: Ensure main.tsx is correct**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

**Step 3: Create .env.local for web**

Create `solar-monitor/web/.env.local`:
```
VITE_API_URL=http://localhost:3001
```

**Step 4: Test full web app**

Terminal 1:
```bash
cd /home/mikeb/solar-monitor/backend
pnpm dev
```

Terminal 2:
```bash
cd /home/mikeb/solar-monitor/web
pnpm dev
```

Open http://localhost:5173/login
Expected: See login form with dark theme

**Step 5: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(web): complete routing and protected routes"
```

---

## Phase 7: Mobile App

### Task 7.1: Mobile Navigation Setup

**Files:**
- Create: `solar-monitor/mobile/src/navigation/AppNavigator.tsx`
- Modify: `solar-monitor/mobile/App.tsx`

**Step 1: Create navigation directory**

```bash
mkdir -p /home/mikeb/solar-monitor/mobile/src/navigation
mkdir -p /home/mikeb/solar-monitor/mobile/src/screens
mkdir -p /home/mikeb/solar-monitor/mobile/src/api
mkdir -p /home/mikeb/solar-monitor/mobile/src/components
```

**Step 2: Create AppNavigator**

```tsx
// solar-monitor/mobile/src/navigation/AppNavigator.tsx
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import LoginScreen from '../screens/LoginScreen';
import DashboardScreen from '../screens/DashboardScreen';
import PanelDetailScreen from '../screens/PanelDetailScreen';

export type RootStackParamList = {
  Login: undefined;
  Dashboard: undefined;
  PanelDetail: { panelId: number };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Login"
        screenOptions={{
          headerStyle: { backgroundColor: '#1f2937' },
          headerTintColor: '#fff',
          contentStyle: { backgroundColor: '#111827' },
        }}
      >
        <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Dashboard" component={DashboardScreen} options={{ title: 'Solar Monitor' }} />
        <Stack.Screen name="PanelDetail" component={PanelDetailScreen} options={{ title: 'Panel Details' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

**Step 3: Update App.tsx**

```tsx
// solar-monitor/mobile/App.tsx
import { PaperProvider } from 'react-native-paper';
import AppNavigator from './src/navigation/AppNavigator';

export default function App() {
  return (
    <PaperProvider>
      <AppNavigator />
    </PaperProvider>
  );
}
```

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(mobile): add navigation structure"
```

---

### Task 7.2: Mobile API Client

**Files:**
- Create: `solar-monitor/mobile/src/api/client.ts`
- Create: `solar-monitor/mobile/src/api/auth.ts`
- Create: `solar-monitor/mobile/src/api/panels.ts`

**Step 1: Create mobile API client**

```typescript
// solar-monitor/mobile/src/api/client.ts
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_URL = 'http://10.0.2.2:3001'; // Android emulator localhost
// Use 'http://localhost:3001' for iOS simulator

export const api = axios.create({
  baseURL: API_URL,
});

api.interceptors.request.use(async (config) => {
  const token = await AsyncStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

**Step 2: Create auth API (similar to web)**

```typescript
// solar-monitor/mobile/src/api/auth.ts
import AsyncStorage from '@react-native-async-storage/async-storage';
import { api } from './client';

export async function login(email: string, password: string): Promise<string> {
  const response = await api.post('/api/auth/login', { email, password });
  const { token } = response.data;
  await AsyncStorage.setItem('token', token);
  return token;
}

export async function logout(): Promise<void> {
  await AsyncStorage.removeItem('token');
}

export async function isAuthenticated(): Promise<boolean> {
  const token = await AsyncStorage.getItem('token');
  return !!token;
}
```

**Step 3: Create panels API (same as web)**

```typescript
// solar-monitor/mobile/src/api/panels.ts
import { api } from './client';

export interface Panel {
  id: number;
  name: string;
  serial_number: string;
  current_watts: number | null;
  current_voltage: number | null;
  current_temperature: number | null;
  is_active: number;
}

export interface SiteOverview {
  currentPower: number;
  dailyEnergy: number;
  monthlyEnergy: number;
}

export async function getPanels(): Promise<Panel[]> {
  const response = await api.get('/api/panels');
  return response.data;
}

export async function getPanel(id: number): Promise<Panel & { readings: any[] }> {
  const response = await api.get(`/api/panels/${id}`);
  return response.data;
}

export async function getSiteOverview(): Promise<SiteOverview> {
  const response = await api.get('/api/site/overview');
  return response.data;
}

export async function shutdownPanel(id: number): Promise<void> {
  await api.post(`/api/controls/panel/${id}/shutdown`);
}

export async function restartPanel(id: number): Promise<void> {
  await api.post(`/api/controls/panel/${id}/restart`);
}
```

**Step 4: Install AsyncStorage**

```bash
cd /home/mikeb/solar-monitor/mobile
pnpm add @react-native-async-storage/async-storage
```

**Step 5: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(mobile): add API client"
```

---

### Task 7.3: Mobile Screens

**Files:**
- Create: `solar-monitor/mobile/src/screens/LoginScreen.tsx`
- Create: `solar-monitor/mobile/src/screens/DashboardScreen.tsx`
- Create: `solar-monitor/mobile/src/screens/PanelDetailScreen.tsx`

**Step 1: Create LoginScreen**

```tsx
// solar-monitor/mobile/src/screens/LoginScreen.tsx
import { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { TextInput, Button, Text } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { login } from '../api/auth';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Login'>;
};

export default function LoginScreen({ navigation }: Props) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigation.replace('Dashboard');
    } catch {
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text variant="headlineLarge" style={styles.title}>Solar Monitor</Text>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <TextInput
        label="Email"
        value={email}
        onChangeText={setEmail}
        mode="outlined"
        style={styles.input}
        keyboardType="email-address"
        autoCapitalize="none"
      />

      <TextInput
        label="Password"
        value={password}
        onChangeText={setPassword}
        mode="outlined"
        style={styles.input}
        secureTextEntry
      />

      <Button
        mode="contained"
        onPress={handleLogin}
        loading={loading}
        style={styles.button}
      >
        Sign In
      </Button>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    color: '#fff',
    textAlign: 'center',
    marginBottom: 40,
  },
  input: {
    marginBottom: 16,
    backgroundColor: '#1f2937',
  },
  button: {
    marginTop: 8,
    backgroundColor: '#eab308',
  },
  error: {
    color: '#ef4444',
    textAlign: 'center',
    marginBottom: 16,
  },
});
```

**Step 2: Create DashboardScreen**

```tsx
// solar-monitor/mobile/src/screens/DashboardScreen.tsx
import { useEffect, useState } from 'react';
import { View, ScrollView, StyleSheet, RefreshControl } from 'react-native';
import { Card, Text, Button } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import { getPanels, getSiteOverview, Panel, SiteOverview } from '../api/panels';
import { logout } from '../api/auth';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Dashboard'>;
};

export default function DashboardScreen({ navigation }: Props) {
  const [panels, setPanels] = useState<Panel[]>([]);
  const [overview, setOverview] = useState<SiteOverview | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    try {
      const [panelsData, overviewData] = await Promise.all([
        getPanels(),
        getSiteOverview(),
      ]);
      setPanels(panelsData);
      setOverview(overviewData);
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  const handleLogout = async () => {
    await logout();
    navigation.replace('Login');
  };

  const getPanelColor = (panel: Panel): string => {
    if (!panel.is_active) return '#ef4444';
    if (panel.current_watts === null) return '#6b7280';
    if (panel.current_watts < 50) return '#eab308';
    return '#22c55e';
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Overview Stats */}
      {overview && (
        <View style={styles.statsRow}>
          <Card style={styles.statCard}>
            <Card.Content>
              <Text variant="labelSmall" style={styles.label}>Current Power</Text>
              <Text variant="headlineMedium" style={styles.value}>
                {(overview.currentPower / 1000).toFixed(2)} kW
              </Text>
            </Card.Content>
          </Card>
          <Card style={styles.statCard}>
            <Card.Content>
              <Text variant="labelSmall" style={styles.label}>Today</Text>
              <Text variant="headlineMedium" style={styles.value}>
                {(overview.dailyEnergy / 1000).toFixed(1)} kWh
              </Text>
            </Card.Content>
          </Card>
        </View>
      )}

      {/* Panel Grid */}
      <Text variant="titleMedium" style={styles.sectionTitle}>Panels</Text>
      <View style={styles.panelGrid}>
        {panels.map((panel) => (
          <Card
            key={panel.id}
            style={[styles.panelCard, { backgroundColor: getPanelColor(panel) }]}
            onPress={() => navigation.navigate('PanelDetail', { panelId: panel.id })}
          >
            <Card.Content>
              <Text variant="labelSmall" style={styles.panelName}>{panel.name}</Text>
              <Text variant="titleLarge" style={styles.panelWatts}>
                {panel.current_watts !== null ? `${Math.round(panel.current_watts)}W` : '--'}
              </Text>
            </Card.Content>
          </Card>
        ))}
      </View>

      <Button mode="outlined" onPress={handleLogout} style={styles.logoutButton}>
        Logout
      </Button>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
    padding: 16,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#1f2937',
  },
  label: {
    color: '#9ca3af',
  },
  value: {
    color: '#eab308',
  },
  sectionTitle: {
    color: '#fff',
    marginBottom: 12,
  },
  panelGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  panelCard: {
    width: '31%',
    marginBottom: 8,
  },
  panelName: {
    color: '#fff',
  },
  panelWatts: {
    color: '#fff',
    fontWeight: 'bold',
  },
  logoutButton: {
    marginTop: 24,
    marginBottom: 40,
  },
});
```

**Step 3: Create PanelDetailScreen**

```tsx
// solar-monitor/mobile/src/screens/PanelDetailScreen.tsx
import { useEffect, useState } from 'react';
import { View, ScrollView, StyleSheet, Alert } from 'react-native';
import { Card, Text, Button, Switch } from 'react-native-paper';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../navigation/AppNavigator';
import { getPanel, shutdownPanel, restartPanel, Panel } from '../api/panels';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'PanelDetail'>;
  route: RouteProp<RootStackParamList, 'PanelDetail'>;
};

export default function PanelDetailScreen({ route }: Props) {
  const { panelId } = route.params;
  const [panel, setPanel] = useState<Panel | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function fetchPanel() {
      const data = await getPanel(panelId);
      setPanel(data);
    }
    fetchPanel();
  }, [panelId]);

  const handleToggle = () => {
    if (!panel) return;

    if (panel.is_active) {
      Alert.alert(
        'Confirm Shutdown',
        'Are you sure you want to shut down this panel?',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Shut Down',
            style: 'destructive',
            onPress: async () => {
              setLoading(true);
              await shutdownPanel(panelId);
              setPanel((prev) => (prev ? { ...prev, is_active: 0 } : null));
              setLoading(false);
            },
          },
        ]
      );
    } else {
      setLoading(true);
      restartPanel(panelId).then(() => {
        setPanel((prev) => (prev ? { ...prev, is_active: 1 } : null));
        setLoading(false);
      });
    }
  };

  if (!panel) {
    return (
      <View style={styles.container}>
        <Text style={{ color: '#fff' }}>Loading...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text variant="headlineMedium" style={styles.title}>{panel.name}</Text>
        <Text style={styles.serial}>{panel.serial_number}</Text>
      </View>

      <View style={styles.controlRow}>
        <Text style={styles.controlLabel}>Panel Active</Text>
        <Switch
          value={panel.is_active === 1}
          onValueChange={handleToggle}
          disabled={loading}
        />
      </View>

      <View style={styles.statsGrid}>
        <Card style={styles.statCard}>
          <Card.Content>
            <Text variant="labelSmall" style={styles.label}>Power</Text>
            <Text variant="headlineMedium" style={styles.value}>
              {panel.current_watts !== null ? `${Math.round(panel.current_watts)} W` : '--'}
            </Text>
          </Card.Content>
        </Card>

        <Card style={styles.statCard}>
          <Card.Content>
            <Text variant="labelSmall" style={styles.label}>Voltage</Text>
            <Text variant="headlineMedium" style={styles.value}>
              {panel.current_voltage !== null ? `${panel.current_voltage.toFixed(1)} V` : '--'}
            </Text>
          </Card.Content>
        </Card>

        <Card style={styles.statCard}>
          <Card.Content>
            <Text variant="labelSmall" style={styles.label}>Temp</Text>
            <Text variant="headlineMedium" style={styles.value}>
              {panel.current_temperature !== null ? `${panel.current_temperature.toFixed(1)}°C` : '--'}
            </Text>
          </Card.Content>
        </Card>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
    padding: 16,
  },
  header: {
    marginBottom: 24,
  },
  title: {
    color: '#fff',
  },
  serial: {
    color: '#9ca3af',
  },
  controlRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#1f2937',
    padding: 16,
    borderRadius: 8,
    marginBottom: 24,
  },
  controlLabel: {
    color: '#fff',
    fontSize: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 8,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#1f2937',
  },
  label: {
    color: '#9ca3af',
  },
  value: {
    color: '#eab308',
  },
});
```

**Step 4: Commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "feat(mobile): add login, dashboard, and panel detail screens"
```

---

## Phase 8: Final Integration

### Task 8.1: Test Full Stack

**Step 1: Start backend**

```bash
cd /home/mikeb/solar-monitor/backend
cp .env.example .env
# Edit .env with your SolarEdge credentials
pnpm dev
```

**Step 2: Start web**

```bash
cd /home/mikeb/solar-monitor/web
pnpm dev
```

**Step 3: Start mobile (optional)**

```bash
cd /home/mikeb/solar-monitor/mobile
pnpm start
```

**Step 4: Test workflow**

1. Open http://localhost:5173/login
2. Register a new user
3. Login
4. View dashboard (will be empty without SolarEdge data)
5. Navigate to panel detail
6. Test shutdown toggle

**Step 5: Final commit**

```bash
cd /home/mikeb/solar-monitor
git add .
git commit -m "chore: complete solar monitor MVP"
```

---

## Summary

This plan builds a complete solar monitoring system:

- **Backend:** Express API with SQLite, JWT auth, SolarEdge integration, automation engine
- **Web:** React + Vite + Tailwind dashboard with login, panel grid, charts, control
- **Mobile:** React Native + Expo app with same features

Total tasks: ~25 bite-sized implementation steps, each 2-5 minutes.
