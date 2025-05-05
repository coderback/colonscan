// app/page.jsx
import { redirect } from 'next/navigation';

export default function Home() {
  // immediately send everyone at “/” to “/login”
  redirect('/login');
}
