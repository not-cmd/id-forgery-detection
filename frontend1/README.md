# Document Forgery Detection Frontend

This is the frontend for the Document Forgery Detection system. It's built with React, TypeScript, and Tailwind CSS.

## Features

- ID card forgery detection
- Signature forgery detection
- Custom model training
- Responsive design

## Setup

1. Make sure you have Node.js installed (v14+ recommended)
2. Navigate to the project directory:
   ```
   cd project
   ```
3. Install dependencies:
   ```
   npm install
   ```
4. Start the development server:
   ```
   npm run dev
   ```

## Building for Production

To build the frontend for production:

```
npm run build
```

The build output will be in the `dist` directory.

## Integration with Backend

The frontend is configured to communicate with the Flask backend running on port 5002. The Vite development server is set up with a proxy configuration to forward API requests to the backend.

## Running the Complete Application

Use the provided `run_app.sh` script in the root directory to start both the frontend and backend together:

```
./run_app.sh
```

This will start the Flask backend on port 5002 and the React frontend on port 3000. 