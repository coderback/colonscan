"use client"
import { useRouter, usePathname } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '../context/AuthContext'
import { Activity, FileText, Video, Dna, LogOut } from 'lucide-react'

export default function Sidebar() {
  const router = useRouter()
  const pathname = usePathname()
  const { logout } = useAuth()

  const navItems = [
    { label: 'Dashboard', href: '/dashboard', icon: Activity },
    { label: 'WSI Classification', href: '/dashboard/wsi', icon: FileText },
    { label: 'Polyp Segmentation', href: '/dashboard/polyp', icon: Video },
    { label: 'Genomic Profile Analysis', href: '/dashboard/genomic', icon: Dna },
  ]

  const handleLogout = async () => {
    await logout()
    router.push('/login')
  }

  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col justify-between shadow-sm">
      <div className="p-6">
        <h1 className="text-xl font-bold text-[#005EB8] mb-8">ColonoScan</h1>
        <nav className="space-y-2">
          {navItems.map(item => {
            const Icon = item.icon
            const isActive = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center px-4 py-3 text-sm font-medium transition-colors duration-200 ${
                  isActive 
                    ? 'bg-[#005EB8]/5 border-r-2 border-[#005EB8] text-[#005EB8]' 
                    : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <Icon className="h-5 w-5 mr-3" />
                {item.label}
              </Link>
            )
          })}
        </nav>
      </div>
      <div className="p-4 border-t border-gray-200">
        <button
          onClick={handleLogout}
          className="w-full flex items-center px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 hover:text-gray-900 transition-colors duration-200"
        >
          <LogOut className="h-5 w-5 mr-3" />
          Log Out
        </button>
      </div>
    </aside>
  )
}