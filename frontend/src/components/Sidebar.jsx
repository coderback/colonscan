"use client"
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '../context/AuthContext'

export default function Sidebar() {
  const router = useRouter()
  const { logout } = useAuth()

  const navItems = [
    { label: 'WSI Classification', href: '/dashboard/wsi' },
    { label: 'Polyp Segmentation', href: '/dashboard/polyp' },
    { label: 'Genomic Profile Analysis', href: '/dashboard/genomic' },
  ]

  const handleLogout = async () => {
    await logout()
    router.push('/login')
  }

  return (
    <aside className="w-64 bg-white border-r flex flex-col justify-between">
      <nav className="mt-10 space-y-2">
        {navItems.map(item => (
          <Link
            key={item.href}
            href={item.href}
            className="block px-4 py-2 hover:bg-gray-100"
          >
            {item.label}
          </Link>
        ))}
      </nav>
      <button
        onClick={handleLogout}
        className="mb-4 mx-4 py-2 bg-red-500 text-white rounded"
      >
        Log Out
      </button>
    </aside>
  )
}