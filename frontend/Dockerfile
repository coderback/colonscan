# frontend/Dockerfile - Development version with hot reloading

FROM node:22.16.0-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Expose the port Next.js listens on
EXPOSE 3000

# Start the Next.js development server
CMD ["npm", "run", "dev"]