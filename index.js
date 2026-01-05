import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import mongoose from "mongoose";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import helmet from "helmet";
import morgan from "morgan";
import rateLimit from "express-rate-limit";
import compression from "compression";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const config = {
  // Server
  PORT: process.env.PORT || 4000,
  NODE_ENV: process.env.NODE_ENV || "development",
  
  // Database
  MONGODB_URI: process.env.MONGODB_URI || "mongodb://localhost:27017/medai",
  
  // JWT
  JWT_SECRET: process.env.JWT_SECRET || "replace_this_with_a_long_secure_secret_in_production",
  JWT_EXPIRATION: process.env.JWT_EXPIRATION || "7d",
  
  // AI Service
  AI_PORT: process.env.AI_PORT || 8000,
  AI_HOST: process.env.AI_HOST || "127.0.0.1",
  
  // Security
  CORS_ORIGIN: process.env.CORS_ORIGIN || "*",
  RATE_LIMIT_WINDOW: parseInt(process.env.RATE_LIMIT_WINDOW) || 15 * 60 * 1000, // 15 minutes
  RATE_LIMIT_MAX: parseInt(process.env.RATE_LIMIT_MAX) || 100,
  
  // File upload
  MAX_FILE_SIZE: parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024, // 10MB
  ALLOWED_FILE_TYPES: [
    'image/jpeg', 
    'image/jpg', 
    'image/png', 
    'image/gif', 
    'image/bmp', 
    'image/tiff',
    'image/webp',
    'application/dicom',
    'application/octet-stream'
  ]
};

// Validate required environment variables
if (config.NODE_ENV === "production") {
  const required = ["MONGODB_URI", "JWT_SECRET"];
  const missing = required.filter(key => !process.env[key]);
  
  if (missing.length > 0) {
    console.error(`❌ Missing required environment variables: ${missing.join(", ")}`);
    process.exit(1);
  }
}

console.log(`🚀 Starting MedAI Server in ${config.NODE_ENV} mode`);

// ====== 1. Start Python AI Backend ====== //
let pythonProcess = null;
const startPythonAI = () => {
  try {
    const venvPythonPath = path.join(__dirname, "venv", "bin", "python3");
    
    // Check if virtual environment exists
    if (!fs.existsSync(path.join(__dirname, "venv"))) {
      console.warn("⚠️  Python virtual environment not found. Skipping AI backend startup.");
      return;
    }
    
    pythonProcess = spawn(venvPythonPath, [
      "-m", "uvicorn", 
      "main:app", 
      "--host", config.AI_HOST,
      "--port", config.AI_PORT.toString(),
      "--reload", config.NODE_ENV === "development" ? "--reload" : ""
    ], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on("data", (data) => {
      console.log(`🤖 [AI]: ${data.toString().trim()}`);
    });
    
    pythonProcess.stderr.on("data", (data) => {
      const message = data.toString().trim();
      if (!message.includes("Uvicorn running") && !message.includes("Application startup complete")) {
        console.error(`🤖 [AI-Error]: ${message}`);
      }
    });
    
    pythonProcess.on("error", (err) => {
      console.error(`❌ Failed to start Python AI: ${err.message}`);
    });
    
    pythonProcess.on("exit", (code, signal) => {
      console.warn(`🤖 Python AI process exited with code ${code}, signal ${signal}`);
    });
    
    console.log(`🤖 Python AI backend starting on http://${config.AI_HOST}:${config.AI_PORT}`);
  } catch (error) {
    console.error("❌ Error starting Python AI:", error.message);
  }
};

// ====== 2. Database ====== //
mongoose.set('strictQuery', false);

// Database connection with retry logic
const connectDB = async (retries = 5, delay = 5000) => {
  for (let i = 0; i < retries; i++) {
    try {
      await mongoose.connect(config.MONGODB_URI, {
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS: 45000,
      });
      console.log("✅ MongoDB connected successfully");
      return true;
    } catch (err) {
      console.error(`❌ MongoDB connection attempt ${i + 1} failed:`, err.message);
      
      if (i < retries - 1) {
        console.log(`⏳ Retrying in ${delay / 1000} seconds...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        if (config.NODE_ENV === "production") {
          console.error("❌ Failed to connect to MongoDB after all retries");
          process.exit(1);
        } else {
          console.warn("⚠️  Continuing without database connection in development mode");
          return false;
        }
      }
    }
  }
};

// User Schema
const UserSchema = new mongoose.Schema({
  email: { 
    type: String, 
    unique: true, 
    required: true, 
    lowercase: true,
    trim: true,
    match: [/^\S+@\S+\.\S+$/, 'Please enter a valid email']
  },
  passwordHash: { type: String, required: true },
  name: { 
    type: String, 
    required: true,
    trim: true,
    minlength: 2,
    maxlength: 50
  },
  role: { 
    type: String, 
    enum: ["user", "doctor", "admin"], 
    default: "user" 
  },
  createdAt: { type: Date, default: Date.now },
  lastLogin: { type: Date },
  isActive: { type: Boolean, default: true }
});

// Indexes
UserSchema.index({ email: 1 });
UserSchema.index({ createdAt: -1 });

const User = mongoose.model("User", UserSchema);

// ====== 3. Express App ====== //
const app = express();

// Security middleware
app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" },
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      scriptSrc: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      imgSrc: ["'self'", "data:", "blob:", "https:"],
      connectSrc: ["'self'", "http://localhost:*", "ws://localhost:*"]
    }
  }
}));

// CORS configuration
const corsOptions = {
  origin: config.CORS_ORIGIN === "*" ? "*" : config.CORS_ORIGIN.split(","),
  credentials: true,
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
  allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With", "x-user-role"],
  exposedHeaders: ["Content-Disposition"]
};

app.use(cors(corsOptions));

// Rate limiting
const limiter = rateLimit({
  windowMs: config.RATE_LIMIT_WINDOW,
  max: config.RATE_LIMIT_MAX,
  message: {
    message: "Too many requests from this IP, please try again later.",
    retryAfter: config.RATE_LIMIT_WINDOW / 1000
  },
  standardHeaders: true,
  legacyHeaders: false
});

app.use(limiter);

// Compression
app.use(compression());

// Logging
const morganFormat = config.NODE_ENV === "production" ? "combined" : "dev";
app.use(morgan(morganFormat, {
  skip: (req, res) => req.path === "/health"
}));

// Body parsing
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// File upload middleware
const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: { fileSize: config.MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    if (config.ALLOWED_FILE_TYPES.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type. Allowed types: ${config.ALLOWED_FILE_TYPES.join(", ")}`));
    }
  }
});

// Utility functions
const generateToken = (user) => {
  return jwt.sign(
    { 
      id: user._id, 
      email: user.email, 
      role: user.role,
      name: user.name 
    },
    config.JWT_SECRET,
    { expiresIn: config.JWT_EXPIRATION }
  );
};

const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.startsWith('Bearer ') 
    ? authHeader.slice(7) 
    : authHeader;
    
  if (!token) {
    return res.status(401).json({ 
      message: "Access denied. No token provided.",
      code: "NO_TOKEN"
    });
  }

  jwt.verify(token, config.JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ 
        message: "Invalid or expired token.",
        code: err.name === "TokenExpiredError" ? "TOKEN_EXPIRED" : "INVALID_TOKEN"
      });
    }
    req.user = user;
    next();
  });
};

// ====== 4. Routes ====== //

// Health check
app.get("/health", async (req, res) => {
  const health = {
    status: "healthy",
    timestamp: new Date().toISOString(),
    service: "MedAI API",
    version: "1.0.0",
    environment: config.NODE_ENV,
    uptime: process.uptime()
  };
  
  // Database status
  try {
    health.database = mongoose.connection.readyState === 1 ? "connected" : "disconnected";
    if (mongoose.connection.readyState === 1) {
      await mongoose.connection.db.admin().ping();
      health.database = "healthy";
    }
  } catch (err) {
    health.database = "unhealthy";
  }
  
  // AI service status
  health.ai_service = pythonProcess && !pythonProcess.killed ? "running" : "stopped";
  
  res.json(health);
});

// API Documentation
app.get("/", (req, res) => {
  res.json({
    message: "MedAI Medical Diagnostic API",
    version: "1.0.0",
    endpoints: {
      auth: {
        register: "POST /auth/register",
        login: "POST /auth/login"
      },
      diagnostics: {
        process: "POST /diagnostics/process (requires auth)"
      },
      system: {
        health: "GET /health",
        docs: "GET /"
      }
    },
    documentation: "https://github.com/yourusername/medai/docs"
  });
});

// Authentication routes
app.post("/auth/register", async (req, res) => {
  try {
    const { email, password, name, role = "user" } = req.body;
    
    // Validation
    if (!email || !password || !name) {
      return res.status(400).json({ 
        message: "Email, password, and name are required.",
        code: "VALIDATION_ERROR"
      });
    }
    
    if (password.length < 8) {
      return res.status(400).json({ 
        message: "Password must be at least 8 characters.",
        code: "PASSWORD_TOO_SHORT"
      });
    }
    
    if (!["user", "doctor", "admin"].includes(role)) {
      return res.status(400).json({ 
        message: "Invalid role specified.",
        code: "INVALID_ROLE"
      });
    }
    
    // Check if user exists
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(409).json({ 
        message: "User with this email already exists.",
        code: "USER_EXISTS"
      });
    }

    // Create user
    const passwordHash = await bcrypt.hash(password, 12);
    const user = new User({ 
      email: email.toLowerCase(), 
      passwordHash, 
      name: name.trim(), 
      role 
    });
    
    await user.save();

    // Generate token
    const token = generateToken(user);
    
    // Update last login
    user.lastLogin = new Date();
    await user.save();

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt
      },
      message: "Registration successful"
    });
    
  } catch (err) {
    console.error("Registration error:", err);
    
    // Handle duplicate key error
    if (err.code === 11000) {
      return res.status(409).json({ 
        message: "User with this email already exists.",
        code: "DUPLICATE_EMAIL"
      });
    }
    
    res.status(500).json({ 
      message: "Error registering user.", 
      code: "REGISTRATION_ERROR",
      error: config.NODE_ENV === "development" ? err.message : undefined
    });
  }
});

app.post("/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ 
        message: "Email and password are required.",
        code: "VALIDATION_ERROR"
      });
    }
    
    const user = await User.findOne({ email: email.toLowerCase() });
    
    if (!user) {
      return res.status(401).json({ 
        message: "Invalid email or password.",
        code: "INVALID_CREDENTIALS"
      });
    }
    
    if (!user.isActive) {
      return res.status(403).json({ 
        message: "Account is deactivated.",
        code: "ACCOUNT_DEACTIVATED"
      });
    }
    
    const validPassword = await bcrypt.compare(password, user.passwordHash);
    
    if (!validPassword) {
      return res.status(401).json({ 
        message: "Invalid email or password.",
        code: "INVALID_CREDENTIALS"
      });
    }
    
    // Update last login
    user.lastLogin = new Date();
    await user.save();
    
    // Generate token
    const token = generateToken(user);
    
    res.json({
      success: true,
      token,
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        lastLogin: user.lastLogin
      },
      message: "Login successful"
    });
    
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ 
      message: "Error during login.", 
      code: "LOGIN_ERROR",
      error: config.NODE_ENV === "development" ? err.message : undefined
    });
  }
});

// Protected routes
app.get("/auth/me", authenticateToken, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select("-passwordHash");
    
    if (!user) {
      return res.status(404).json({ 
        message: "User not found.",
        code: "USER_NOT_FOUND"
      });
    }
    
    res.json({
      success: true,
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt,
        lastLogin: user.lastLogin
      }
    });
    
  } catch (err) {
    console.error("Profile fetch error:", err);
    res.status(500).json({ 
      message: "Error fetching user profile.", 
      code: "PROFILE_FETCH_ERROR"
    });
  }
});

// Diagnostics route
app.post("/diagnostics/process", authenticateToken, upload.single('file'), async (req, res) => {
  try {
    console.log("Diagnostics request from user:", req.user.email);
    
    if (!req.file) {
      return res.status(400).json({ 
        message: "No image file uploaded.",
        code: "NO_FILE_UPLOADED",
        hint: "Ensure the file is sent with key 'file' in multipart/form-data"
      });
    }

    // Validate file size
    if (req.file.size > config.MAX_FILE_SIZE) {
      return res.status(400).json({
        message: `File too large. Maximum size is ${config.MAX_FILE_SIZE / (1024 * 1024)}MB`,
        code: "FILE_TOO_LARGE"
      });
    }

    // Prepare form data for AI service
    const formData = new FormData();
    formData.append('file', req.file.buffer, { 
      filename: req.file.originalname || 'uploaded_image.jpg', 
      contentType: req.file.mimetype 
    });

    // Call AI service
    const aiServiceUrl = `http://${config.AI_HOST}:${config.AI_PORT}/diagnostics/process`;
    
    console.log(`Calling AI service at ${aiServiceUrl}`);
    
    const response = await axios.post(aiServiceUrl, formData, {
      headers: { 
        ...formData.getHeaders(),
        'x-user-id': req.user.id,
        'x-user-role': req.user.role,
        'x-user-email': req.user.email
      },
      params: { 
        type: req.query.type || "xray",
        timestamp: new Date().toISOString()
      },
      timeout: 60000 // 60 second timeout for AI processing
    });

    // Log successful processing
    console.log(`Diagnostics processed for ${req.user.email}:`, response.data?.diagnosis || "Unknown");

    res.json({
      success: true,
      ...response.data,
      processedAt: new Date().toISOString(),
      requestedBy: {
        id: req.user.id,
        email: req.user.email,
        role: req.user.role
      }
    });
    
  } catch (error) {
    console.error("Diagnostics processing error:", error.message);
    
    // Handle specific errors
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.status(503).json({ 
        message: "AI Service is currently unavailable", 
        code: "AI_SERVICE_UNAVAILABLE",
        detail: "The diagnostic AI service is not running or unreachable",
        hint: "Check if the Python AI server is running on port " + config.AI_PORT
      });
    }
    
    if (error.response) {
      // Forward AI service error with status code
      return res.status(error.response.status).json({
        message: "AI Service error",
        code: "AI_SERVICE_ERROR",
        detail: error.response.data,
        status: error.response.status
      });
    }
    
    if (error.code === 'ETIMEDOUT') {
      return res.status(504).json({ 
        message: "AI Service timeout", 
        code: "AI_SERVICE_TIMEOUT",
        detail: "The diagnostic service took too long to respond"
      });
    }
    
    // Generic error
    res.status(500).json({ 
      message: "Error processing diagnostics", 
      code: "DIAGNOSTICS_ERROR",
      error: config.NODE_ENV === "development" ? error.message : undefined,
      stack: config.NODE_ENV === "development" ? error.stack : undefined
    });
  }
});

// Test endpoint (development only)
if (config.NODE_ENV === "development") {
  app.post("/test/upload", upload.single('file'), (req, res) => {
    if (!req.file) {
      return res.status(400).json({ 
        message: "No file uploaded",
        receivedHeaders: req.headers,
        receivedBody: req.body
      });
    }
    
    res.json({
      message: "File upload test successful",
      file: {
        originalname: req.file.originalname,
        mimetype: req.file.mimetype,
        size: req.file.size,
        encoding: req.file.encoding,
        fieldname: req.file.fieldname
      },
      headers: req.headers
    });
  });
}

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  
  // Multer errors
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        message: `File too large. Maximum size is ${config.MAX_FILE_SIZE / (1024 * 1024)}MB`,
        code: "FILE_TOO_LARGE"
      });
    }
    
    return res.status(400).json({
      message: "File upload error",
      code: "UPLOAD_ERROR",
      detail: err.message
    });
  }
  
  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    return res.status(401).json({
      message: "Invalid token",
      code: "INVALID_TOKEN"
    });
  }
  
  // Validation errors
  if (err.name === 'ValidationError') {
    return res.status(400).json({
      message: "Validation error",
      code: "VALIDATION_ERROR",
      errors: Object.values(err.errors).map(e => e.message)
    });
  }
  
  // Default error
  res.status(err.status || 500).json({
    message: err.message || "Internal server error",
    code: err.code || "INTERNAL_ERROR",
    stack: config.NODE_ENV === "development" ? err.stack : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    message: "Route not found",
    code: "ROUTE_NOT_FOUND",
    requestedUrl: req.originalUrl,
    method: req.method
  });
});

// ====== 5. Startup ====== //
const startServer = async () => {
  try {
    // Connect to database
    await connectDB();
    
    // Start Python AI backend (non-blocking)
    startPythonAI();
    
    // Start Express server
    app.listen(config.PORT, () => {
      console.log(`🚀 Server running on port ${config.PORT}`);
      console.log(`🌐 Environment: ${config.NODE_ENV}`);
      console.log(`📡 Health check: http://localhost:${config.PORT}/health`);
      console.log(`🤖 AI Service: http://${config.AI_HOST}:${config.AI_PORT}`);
      console.log(`🔌 MongoDB: ${mongoose.connection.readyState === 1 ? 'Connected' : 'Not connected'}`);
      
      if (config.NODE_ENV === "development") {
        console.log(`🧪 Test upload: POST http://localhost:${config.PORT}/test/upload`);
      }
    });
    
  } catch (error) {
    console.error("❌ Failed to start server:", error);
    process.exit(1);
  }
};

// Graceful shutdown
const shutdown = async (signal) => {
  console.log(`\n🛑 Received ${signal}. Starting graceful shutdown...`);
  
  // Close server
  if (app.server) {
    app.server.close(() => {
      console.log("✅ HTTP server closed");
    });
  }
  
  // Kill Python process
  if (pythonProcess && !pythonProcess.killed) {
    pythonProcess.kill('SIGTERM');
    console.log("✅ Python AI process terminated");
  }
  
  // Close MongoDB connection
  try {
    await mongoose.connection.close();
    console.log("✅ MongoDB connection closed");
  } catch (err) {
    console.error("❌ Error closing MongoDB:", err);
  }
  
  // Exit
  setTimeout(() => {
    console.log("👋 Shutdown complete");
    process.exit(0);
  }, 1000);
};

// Handle shutdown signals
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

// Handle unhandled rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit in production, just log
  if (config.NODE_ENV === "development") {
    process.exit(1);
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
  // Don't exit in production, just log
  if (config.NODE_ENV === "development") {
    process.exit(1);
  }
});

// Start the server
startServer();

// Export for testing
export { app, config, connectDB };