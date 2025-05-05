"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { useAuth } from "@/context/AuthContext"

export default function DashboardLayout({ children }) {
  const { token, logout } = useAuth()
  const router = useRouter()

  // redirect if not logged in
  useEffect(() => {
    if (!token) {
      router.replace("/login")
    }
  }, [token, router])

  // or show a spinner while we check
  if (!token) {
    return null
  }

  return (
    <div className="flex h-screen">
      <aside className="w-64 bg-white border-r flex flex-col justify-between">
        <nav className="mt-10 space-y-2">
          <Link
            href="/dashboard/wsi"
            className="block px-4 py-2 hover:bg-gray-100"
          >
            WSI Classification
          </Link>
          <Link
            href="/dashboard/polyp"
            className="block px-4 py-2 hover:bg-gray-100"
          >
            Polyp Segmentation
          </Link>
          <Link
            href="/dashboard/genomic"
            className="block px-4 py-2 hover:bg-gray-100"
          >
            Genomic Profile Analysis
          </Link>
        </nav>
        <button
          className="mb-4 mx-4 py-2 bg-red-500 text-white rounded"
          onClick={() => {
            logout()
            router.push("/login")
          }}
        >
          Log Out
        </button>
      </aside>

      <main className="flex-1 overflow-auto bg-gray-50">
        {children}
      </main>
    </div>
  )
}
