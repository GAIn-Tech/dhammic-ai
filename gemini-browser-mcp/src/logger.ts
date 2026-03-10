export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: string;
  data?: unknown;
}

export const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

function serializeData(data: unknown): unknown {
  if (data instanceof Error) {
    return {
      name: data.name,
      message: data.message,
      stack: data.stack,
    };
  }
  return data;
}

export interface Logger {
  debug(message: string, data?: unknown): void;
  info(message: string, data?: unknown): void;
  warn(message: string, data?: unknown): void;
  error(message: string, data?: unknown): void;
}

export function createLogger(minLevel: LogLevel): Logger {
  const minLevelNum = LOG_LEVELS[minLevel];

  function write(level: LogLevel, message: string, data?: unknown): void {
    if (LOG_LEVELS[level] < minLevelNum) {
      return;
    }

    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date().toISOString(),
      ...(data !== undefined ? { data: serializeData(data) } : {}),
    };

    process.stderr.write(JSON.stringify(entry) + '\n');
  }

  return {
    debug: (message: string, data?: unknown) => write('debug', message, data),
    info: (message: string, data?: unknown) => write('info', message, data),
    warn: (message: string, data?: unknown) => write('warn', message, data),
    error: (message: string, data?: unknown) => write('error', message, data),
  };
}

const envLevel = process.env['LOG_LEVEL'] ?? 'info';
const validLevel: LogLevel =
  envLevel in LOG_LEVELS ? (envLevel as LogLevel) : 'info';

if (envLevel !== validLevel) {
  process.stderr.write(JSON.stringify({ level: 'warn', message: `Invalid LOG_LEVEL "${envLevel}", falling back to "info"`, timestamp: new Date().toISOString() }) + '\n');
}

export const logger: Logger = createLogger(validLevel);
