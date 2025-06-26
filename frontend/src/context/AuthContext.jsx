'use client'

import { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'

// 1) Create the context (default values only, will be overwritten by the provider)
const AuthContext = createContext({
  token: null,
  login: async () => {},
  signup: async () => {},
  logout: () => {}
})

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null)

  // on mount, pull any saved token
  useEffect(() => {
    const saved = sessionStorage.getItem('authToken')
    if (saved) {
      setToken(saved)
      axios.defaults.headers.common['Authorization'] = `Token ${saved}`
    }
  }, [])

  // login(): call your DRF login endpoint and persist
  async function login(username, password) {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const { data } = await axios.post(`${API_BASE}/api/auth/login`, { username, password })
    const t = data.token
    setToken(t)
    sessionStorage.setItem('authToken', t)
    axios.defaults.headers.common['Authorization'] = `Token ${t}`
    return data
  }

  // signup(): register new user and automatically log them in
  async function signup(userData) {
    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const { data } = await axios.post(`${API_BASE}/api/auth/signup`, userData)
    const t = data.token
    setToken(t)
    sessionStorage.setItem('authToken', t)
    axios.defaults.headers.common['Authorization'] = `Token ${t}`
    return data
  }

  // logout(): clear everything
  function logout() {
    setToken(null)
    sessionStorage.removeItem('authToken')
    delete axios.defaults.headers.common['Authorization']
  }

  return (
    <AuthContext.Provider value={{ token, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// convenience hook for consuming
export function useAuth() {
  return useContext(AuthContext)
}
