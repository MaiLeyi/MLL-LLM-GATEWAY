'use strict';

const { app, BrowserWindow, dialog, nativeTheme, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');

const ROOT = path.join(__dirname, '..');
const PORT = Number(process.env.MLL_PORT || 8000);
const HOST = '127.0.0.1';
const APP_URL = `http://${HOST}:${PORT}`;

let backend = null;
let win = null;
let quitting = false;

function portOpen(port) {
  return new Promise((resolve) => {
    const socket = net.connect({ port, host: HOST });
    const done = (value) => { socket.destroy(); resolve(value); };
    socket.setTimeout(800);
    socket.once('connect', () => done(true));
    socket.once('timeout', () => done(false));
    socket.once('error', () => done(false));
  });
}

async function waitForServer(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await portOpen(PORT)) return true;
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

function pythonPath() {
  const venvPython = path.join(ROOT, '.venv', 'bin', process.platform === 'win32' ? 'python.exe' : 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  return process.platform === 'win32' ? 'python' : 'python3';
}

function backendLaunch() {
  if (app.isPackaged) {
    return {
      command: path.join(process.resourcesPath, 'backend', process.platform === 'win32' ? 'mll-backend.exe' : 'mll-backend'),
      args: [],
      cwd: process.resourcesPath,
    };
  }
  return { command: pythonPath(), args: ['main.py'], cwd: ROOT };
}

function startBackend() {
  const { command, args, cwd } = backendLaunch();
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONPATH: ROOT },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  child.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  child.on('error', (err) => {
    dialog.showErrorBox('无法启动后端', `启动 Python 服务失败：${err.message}`);
    app.quit();
  });
  child.on('exit', (code) => {
    if (backend === child) backend = null;
    if (!quitting && win && code !== 0) {
      dialog.showErrorBox('后端已退出', `Python 服务异常退出（code ${code}），请检查终端日志。`);
      app.quit();
    }
  });
  return child;
}

function stopBackend() {
  if (!backend) return;
  const child = backend;
  backend = null;
  child.kill('SIGTERM');
  setTimeout(() => {
    try { child.kill('SIGKILL'); } catch { /* 已退出 */ }
  }, 3000).unref();
}

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: 'MLL Gateway',
    backgroundColor: nativeTheme.shouldUseDarkColors ? '#111112' : '#ffffff',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.loadURL(APP_URL);
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.on('closed', () => { win = null; });
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.focus();
  });

  app.whenReady().then(async () => {
    // 已有服务在跑就直接复用，避免端口冲突
    if (!(await portOpen(PORT))) backend = startBackend();

    if (!(await waitForServer())) {
      dialog.showErrorBox('无法启动后端', `端口 ${PORT} 在 30 秒内未就绪，请检查 Python 依赖与 .env 配置。`);
      quitting = true;
      stopBackend();
      app.quit();
      return;
    }

    createWindow();
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on('before-quit', () => {
    quitting = true;
    stopBackend();
  });

  app.on('window-all-closed', () => app.quit());
}
