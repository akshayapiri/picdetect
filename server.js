#!/usr/bin/env node
/**
 * Simple HTTP server to run PicDetect locally
 * This solves CORS issues when running from file:// protocol
 * 
 * Usage:
 *     node server.js
 * 
 * Then open: http://localhost:8000
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8000;

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

const server = http.createServer((req, res) => {
    console.log(`${req.method} ${req.url}`);

    // Add CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    let filePath = '.' + req.url;
    if (filePath === './') {
        filePath = './index.html';
    }

    const extname = String(path.extname(filePath)).toLowerCase();
    const contentType = mimeTypes[extname] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 - File Not Found</h1>', 'utf-8');
            } else {
                res.writeHead(500);
                res.end(`Server Error: ${error.code}`, 'utf-8');
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log(`🚀 PicDetect server running at http://localhost:${PORT}`);
    console.log('📝 Open this URL in your browser');
    console.log('⏹️  Press Ctrl+C to stop the server\n');
    
    // Try to open browser automatically
    const { exec } = require('child_process');
    const platform = process.platform;
    let command;
    
    if (platform === 'darwin') {
        command = 'open';
    } else if (platform === 'win32') {
        command = 'start';
    } else {
        command = 'xdg-open';
    }
    
    exec(`${command} http://localhost:${PORT}`, (error) => {
        if (error) {
            // Browser opening failed, that's okay
        }
    });
});


