  // src/components/Sidebar.jsx
  import { NavLink } from "react-router-dom";
  import { FiHome, FiImage, FiVideo, FiMap } from "react-icons/fi";

  export default function Sidebar() {
    const linkBase =
      "flex items-center gap-3 px-4 py-2 rounded-lg text-sm font-medium transition-colors";
    const linkNormal =
      "text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700";
    const linkActive =
      "bg-blue-500 text-white shadow-md hover:bg-blue-600";

    return (
      <aside className="w-64 h-screen fixed left-0 top-0 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col">
        <div className="px-6 py-5 border-b border-gray-200 dark:border-gray-800">
          <h1 className="text-xl font-bold tracking-tight">
            Bitirme Projesi
          </h1>
          <p className="text-xs text-gray-500 mt-1">
            Toplu Taşıma Yoğunluk Analizi
          </p>
        </div>

        <nav className="flex-1 px-4 py-4 space-y-2">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `${linkBase} ${isActive ? linkActive : linkNormal}`
            }
          >
            <FiHome />
            <span>Dashboard</span>
          </NavLink>

          <NavLink
            to="/image"
            className={({ isActive }) =>
              `${linkBase} ${isActive ? linkActive : linkNormal}`
            }
          >
            <FiImage />
            <span>Fotoğraf Analizi</span>
          </NavLink>

          <NavLink
            to="/video"
            className={({ isActive }) =>
              `${linkBase} ${isActive ? linkActive : linkNormal}`
            }
          >
            <FiVideo />
            <span>Video Analizi</span>
          </NavLink>

        <NavLink
    to="/map"
    className={({ isActive }) =>
      `${linkBase} ${isActive ? linkActive : linkNormal}`
    }
  >
    <FiMap />
    <span>Durak Haritası</span>
  </NavLink>

        </nav>

        <div className="px-6 py-4 text-xs text-gray-400 border-t border-gray-200 dark:border-gray-800">
          © 2025 Sinan Türdü
        </div>
      </aside>
    );
  }
