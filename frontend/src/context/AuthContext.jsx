'use client'

import { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'

// 1) Create the context (default values only, will be overwritten by the provider)
const AuthContext = createContext({
  token: null,
  login: async () => {},
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
    const { data } = await axios.post('/api/auth/login/', { username, password })
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
    <AuthContext.Provider value={{ token, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// convenience hook for consuming
export function useAuth() {
  return useContext(AuthContext)
}
