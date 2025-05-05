# frontend/Dockerfile

# 1) Install dependencies
FROM node:22.15.0-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install

# 2) Build the Next.js app
FROM deps AS builder
WORKDIR /app
COPY . .
RUN npm run build

# 3) Production image
FROM node:22.15.0-alpine AS runner
WORKDIR /app

# Copy only the built output and runtime deps
COPY --from=deps    /app/node_modules ./node_modules
COPY --from=builder /app/.next          ./.next
COPY --from=builder /app/public         ./public
COPY --from=builder /app/src/ ./src/
COPY --from=builder /app/package.json   ./
COPY --from=builder /app/package-lock.json   ./
COPY --from=builder /app/jsconfig.json   ./
COPY --from=builder /app/eslint.config.mjs   ./
COPY --from=builder /app/next.config.js   ./
COPY --from=builder /app/postcss.config.mjs   ./
COPY --from=builder /app/.env.local   ./

# Expose the port Next.js listens on
EXPOSE 3000

# Start the Next.js server
CMD ["npm", "run", "dev"]